// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda.h>

#include <cstdint>

namespace holoscan::ops {

enum class DrmBufferAccess {
  kCudaDmabuf,
  kCudaHostMapped,
  kHostMapped,
};

/**
 * @brief DRM dumb framebuffer exposed through the fastest CUDA path available.
 *
 * Thor can import the exported DMABUF directly. On earlier Jetson platforms,
 * including Orin, the buffer is CPU-mapped and CUDA registration is attempted.
 * A plain CPU mapping remains available as a portable fallback.
 */
class DrmBuffer final {
 public:
  DrmBuffer(int drm_fd, int width, int height);
  ~DrmBuffer();

  DrmBuffer(const DrmBuffer&) = delete;
  DrmBuffer& operator=(const DrmBuffer&) = delete;

  [[nodiscard]] std::uint32_t framebuffer() const noexcept { return framebuffer_; }
  [[nodiscard]] std::uint32_t pitch() const noexcept { return pitch_; }
  [[nodiscard]] DrmBufferAccess access() const noexcept { return access_; }
  [[nodiscard]] const char* access_name() const noexcept;
  [[nodiscard]] bool cuda_accessible() const noexcept { return device_pointer_ != nullptr; }
  [[nodiscard]] std::uint8_t* device_pointer() const noexcept { return device_pointer_; }
  [[nodiscard]] std::uint8_t* host_pointer() const noexcept { return host_pointer_; }

 private:
  void map_for_host();
  void try_register_host_mapping_with_cuda() noexcept;
  void reset() noexcept;

  int drm_fd_{-1};
  std::uint32_t handle_{};
  std::uint32_t pitch_{};
  std::uint64_t size_{};
  std::uint32_t framebuffer_{};
  int dmabuf_fd_{-1};
  CUexternalMemory external_memory_{};
  std::uint8_t* device_pointer_{};
  std::uint8_t* host_pointer_{};
  bool host_registered_{};
  DrmBufferAccess access_{DrmBufferAccess::kHostMapped};
};

}  // namespace holoscan::ops
