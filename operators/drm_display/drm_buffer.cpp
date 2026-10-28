// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "drm_display/drm_buffer.hpp"

#include <drm.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <xf86drmMode.h>

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <sstream>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace holoscan::ops {
namespace {

[[nodiscard]] std::string errno_message(std::string_view operation) {
  std::ostringstream message;
  message << operation << " failed: " << std::strerror(errno);
  return message.str();
}

[[nodiscard]] std::string cuda_driver_message(std::string_view operation, CUresult status) {
  const char* error_name = nullptr;
  const char* error_string = nullptr;
  static_cast<void>(cuGetErrorName(status, &error_name));
  static_cast<void>(cuGetErrorString(status, &error_string));
  std::ostringstream message;
  message << operation << " failed: " << (error_name != nullptr ? error_name : "unknown")
          << " (" << (error_string != nullptr ? error_string : "unknown") << ')';
  return message.str();
}

void ensure_cuda_primary_context() noexcept {
  static std::once_flag once;
  static CUcontext primary_context = nullptr;
  std::call_once(once, [] {
    if (cuInit(0) != CUDA_SUCCESS) return;
    CUdevice device{};
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return;
    static_cast<void>(cuDevicePrimaryCtxRetain(&primary_context, device));
  });
  if (primary_context != nullptr) {
    static_cast<void>(cuCtxSetCurrent(primary_context));
  }
}

}  // namespace

DrmBuffer::DrmBuffer(int drm_fd, int width, int height) : drm_fd_(drm_fd) {
  try {
    drm_mode_create_dumb request{};
    request.width = static_cast<std::uint32_t>(width);
    request.height = static_cast<std::uint32_t>(height);
    request.bpp = 32U;
    if (ioctl(drm_fd_, DRM_IOCTL_MODE_CREATE_DUMB, &request) != 0) {
      throw std::runtime_error(errno_message("DRM_IOCTL_MODE_CREATE_DUMB"));
    }
    handle_ = request.handle;
    pitch_ = request.pitch;
    size_ = request.size;

    if (drmModeAddFB(drm_fd_,
                     static_cast<std::uint32_t>(width),
                     static_cast<std::uint32_t>(height),
                     24,
                     32,
                     pitch_,
                     handle_,
                     &framebuffer_) != 0) {
      throw std::runtime_error(errno_message("drmModeAddFB"));
    }

    drm_prime_handle prime{};
    prime.handle = handle_;
    prime.flags = DRM_CLOEXEC | DRM_RDWR;
    if (ioctl(drm_fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) != 0) {
      throw std::runtime_error(errno_message("DRM_IOCTL_PRIME_HANDLE_TO_FD"));
    }
    dmabuf_fd_ = prime.fd;

    ensure_cuda_primary_context();

    CUDA_EXTERNAL_MEMORY_HANDLE_DESC memory_desc{};
    memory_desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_DMABUF_FD;
    memory_desc.handle.fd = dmabuf_fd_;
    memory_desc.size = size_;
    const CUresult imported = cuImportExternalMemory(&external_memory_, &memory_desc);
    if (imported == CUDA_ERROR_NOT_SUPPORTED || imported == CUDA_ERROR_INVALID_CONTEXT) {
      // CUDA DMABUF import on Jetson starts with Thor. Orin uses the mapped
      // framebuffer path below and avoids a separate staging allocation.
      map_for_host();
      try_register_host_mapping_with_cuda();
      return;
    }
    if (imported != CUDA_SUCCESS) {
      throw std::runtime_error(cuda_driver_message("cuImportExternalMemory(DMABUF)", imported));
    }

    CUDA_EXTERNAL_MEMORY_BUFFER_DESC buffer_desc{};
    buffer_desc.size = size_;
    CUdeviceptr mapped_pointer = 0U;
    const CUresult mapped =
        cuExternalMemoryGetMappedBuffer(&mapped_pointer, external_memory_, &buffer_desc);
    if (mapped != CUDA_SUCCESS) {
      throw std::runtime_error(
          cuda_driver_message("cuExternalMemoryGetMappedBuffer(DMABUF)", mapped));
    }
    device_pointer_ = reinterpret_cast<std::uint8_t*>(mapped_pointer);
    access_ = DrmBufferAccess::kCudaDmabuf;
  } catch (...) {
    reset();
    throw;
  }
}

DrmBuffer::~DrmBuffer() { reset(); }

const char* DrmBuffer::access_name() const noexcept {
  switch (access_) {
    case DrmBufferAccess::kCudaDmabuf:
      return "CUDA DMABUF import";
    case DrmBufferAccess::kCudaHostMapped:
      return "CUDA-registered DRM mmap";
    case DrmBufferAccess::kHostMapped:
      return "DRM mmap copy fallback";
  }
  return "unknown";
}

void DrmBuffer::map_for_host() {
  drm_mode_map_dumb request{};
  request.handle = handle_;
  if (ioctl(drm_fd_, DRM_IOCTL_MODE_MAP_DUMB, &request) != 0) {
    throw std::runtime_error(errno_message("DRM_IOCTL_MODE_MAP_DUMB"));
  }

  void* const mapping = mmap(nullptr,
                             static_cast<std::size_t>(size_),
                             PROT_READ | PROT_WRITE,
                             MAP_SHARED,
                             drm_fd_,
                             static_cast<off_t>(request.offset));
  if (mapping == MAP_FAILED) {
    throw std::runtime_error(errno_message("mmap(DRM dumb buffer)"));
  }
  host_pointer_ = static_cast<std::uint8_t*>(mapping);
  std::memset(host_pointer_, 0, static_cast<std::size_t>(size_));
  access_ = DrmBufferAccess::kHostMapped;
}

void DrmBuffer::try_register_host_mapping_with_cuda() noexcept {
  if (host_pointer_ == nullptr) {
    return;
  }

  // Register the DRM mapping so CUDA can transfer directly into scanout memory
  // without allocating or copying through a pageable host staging image.
  CUresult registered = cuMemHostRegister(
      host_pointer_,
      static_cast<std::size_t>(size_),
      CU_MEMHOSTREGISTER_DEVICEMAP | CU_MEMHOSTREGISTER_IOMEMORY);
  if (registered != CUDA_SUCCESS) {
    // NVIDIA DRM can expose a dumb buffer as coherent system memory instead of
    // I/O memory, depending on the Tegra driver version.
    registered = cuMemHostRegister(host_pointer_,
                                   static_cast<std::size_t>(size_),
                                   CU_MEMHOSTREGISTER_DEVICEMAP);
  }
  if (registered != CUDA_SUCCESS) {
    return;
  }
  host_registered_ = true;

  CUdeviceptr mapped_pointer = 0U;
  if (cuMemHostGetDevicePointer(&mapped_pointer, host_pointer_, 0U) != CUDA_SUCCESS) {
    static_cast<void>(cuMemHostUnregister(host_pointer_));
    host_registered_ = false;
    return;
  }
  device_pointer_ = reinterpret_cast<std::uint8_t*>(mapped_pointer);
  access_ = DrmBufferAccess::kCudaHostMapped;
}

void DrmBuffer::reset() noexcept {
  // The device pointer aliases either external memory or registered host memory;
  // it must not be released with cudaFree().
  device_pointer_ = nullptr;
  if (external_memory_ != nullptr) {
    static_cast<void>(cuDestroyExternalMemory(external_memory_));
    external_memory_ = nullptr;
  }
  if (host_registered_ && host_pointer_ != nullptr) {
    static_cast<void>(cuMemHostUnregister(host_pointer_));
    host_registered_ = false;
  }
  if (host_pointer_ != nullptr) {
    static_cast<void>(munmap(host_pointer_, static_cast<std::size_t>(size_)));
    host_pointer_ = nullptr;
  }
  if (dmabuf_fd_ >= 0) {
    static_cast<void>(close(dmabuf_fd_));
    dmabuf_fd_ = -1;
  }
  if (drm_fd_ >= 0 && framebuffer_ != 0U) {
    static_cast<void>(drmModeRmFB(drm_fd_, framebuffer_));
    framebuffer_ = 0U;
  }
  if (drm_fd_ >= 0 && handle_ != 0U) {
    drm_mode_destroy_dumb request{};
    request.handle = handle_;
    static_cast<void>(ioctl(drm_fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &request));
    handle_ = 0U;
  }
  pitch_ = 0U;
  size_ = 0U;
  access_ = DrmBufferAccess::kHostMapped;
}

}  // namespace holoscan::ops
