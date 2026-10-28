// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "drm_bgra_display/drm_bgra_display.hpp"

#include <cuda_runtime_api.h>
#include <poll.h>
#include <xf86drm.h>
#include <xf86drmMode.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "drm_display/drm_buffer.hpp"
#include "drm_display/drm_device.hpp"
#include "flow_trace/flow_trace.hpp"
#include "operator_utils/tensor_utils.hpp"

namespace holoscan::ops {
namespace {

using holoscan::ops::kUInt8Dtype;
using holoscan::ops::same_dtype;
using holoscan::ops::shape_equals;

[[nodiscard]] holoscan::expected<void, holoscan::Error> failure(
    holoscan::ErrorCode code, std::string message) {
  return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
}

[[nodiscard]] std::string errno_message(std::string_view operation) {
  std::ostringstream message;
  message << operation << " failed: " << std::strerror(errno);
  return message.str();
}

[[nodiscard]] std::string cuda_message(std::string_view operation, cudaError_t status) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(status);
  return message.str();
}

}  // namespace

struct DrmBgraDisplayOp::Impl {
  Impl(int width,
       int height,
       std::string device,
       std::uint32_t requested_connector,
       std::string requested_mode)
      : drm(width,
            height,
            std::move(device),
            requested_connector,
            std::move(requested_mode)) {
    event_context.version = DRM_EVENT_CONTEXT_VERSION;
    event_context.page_flip_handler = &Impl::page_flip_handler;
  }

  ~Impl() { cleanup(); }

  static void page_flip_handler(int,
                                unsigned int,
                                unsigned int,
                                unsigned int,
                                void* data) {
    auto* self = static_cast<Impl*>(data);
    self->flip_pending = false;
    self->current_buffer = self->pending_buffer;
  }

  void start() {
    cleanup();
    try {
      drm.start();
      const auto& selected_mode = drm.mode();
      for (auto& buffer : buffers) {
        buffer = std::make_unique<DrmBuffer>(
            drm.fd(), selected_mode.hdisplay, selected_mode.vdisplay);
      }
      drm.set_framebuffer(buffers.front()->framebuffer());

      std::fprintf(stderr,
                   "[drm_bgra_display] device=%s connector=%u crtc=%u "
                   "mode=%dx%d@%d memory=%s\n",
                   drm.path().c_str(),
                   drm.connector_id(),
                   drm.crtc_id(),
                   selected_mode.hdisplay,
                   selected_mode.vdisplay,
                   selected_mode.vrefresh,
                   buffers.front()->access_name());
    } catch (const std::exception& exception) {
      std::fprintf(stderr,
                   "[drm_bgra_display] start failed on %s: %s\n",
                   drm.path().c_str(),
                   exception.what());
      cleanup();
      throw;
    } catch (...) {
      std::fprintf(stderr,
                   "[drm_bgra_display] start failed on %s: unknown error\n",
                   drm.path().c_str());
      cleanup();
      throw;
    }
  }

  void stop() { cleanup(); }

  void wait_for_flip() {
    while (flip_pending) {
      pollfd descriptor{};
      descriptor.fd = drm.fd();
      descriptor.events = POLLIN;
      int result = 0;
      do {
        result = poll(&descriptor, 1, 2000);
      } while (result < 0 && errno == EINTR);
      if (result == 0) {
        throw std::runtime_error("timed out waiting for DRM page-flip event");
      }
      if (result < 0) {
        throw std::runtime_error(errno_message("poll DRM page-flip"));
      }
      if (drmHandleEvent(drm.fd(), &event_context) != 0) {
        throw std::runtime_error(errno_message("drmHandleEvent"));
      }
    }
  }

  void cleanup() noexcept {
    if (flip_pending) {
      try {
        wait_for_flip();
      } catch (const std::exception&) {
        // The device may already be disconnected during shutdown. Continue
        // cleanup and let the kernel reclaim the active DRM resources.
        flip_pending = false;
      }
    }

    if (!drm.is_open()) {
      for (auto& buffer : buffers) {
        buffer.reset();
      }
      return;
    }

    // Restore the saved CRTC while our current framebuffer still exists.
    drm.restore_crtc();
    for (auto& buffer : buffers) {
      buffer.reset();
    }
    drm.close();

    flip_pending = false;
    current_buffer = 0U;
    pending_buffer = 0U;
  }

  DrmDevice drm;
  std::array<std::unique_ptr<DrmBuffer>, 2U> buffers{};
  drmEventContext event_context{};
  std::uint32_t current_buffer{};
  std::uint32_t pending_buffer{};
  bool flip_pending{};
};

DrmBgraDisplayOp::DrmBgraDisplayOp(int width,
                           int height,
                           std::string device,
                           std::uint32_t connector_id,
                           std::string mode)
    : width_(width), height_(height) {
  if (width_ <= 0 || height_ <= 0) {
    throw std::invalid_argument("DRM display width and height must be positive");
  }
  const auto unsigned_width = static_cast<std::size_t>(width_);
  const auto unsigned_height = static_cast<std::size_t>(height_);
  if (unsigned_width > std::numeric_limits<std::size_t>::max() / 4U ||
      unsigned_height > std::numeric_limits<std::size_t>::max() / (unsigned_width * 4U)) {
    throw std::invalid_argument("DRM BGRA frame byte count overflows size_t");
  }
  frame_bytes_ = unsigned_width * unsigned_height * 4U;
  impl_ = std::make_unique<Impl>(
      width_, height_, std::move(device), connector_id, std::move(mode));
}

DrmBgraDisplayOp::~DrmBgraDisplayOp() = default;

void DrmBgraDisplayOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt8Dtype,
              .rank = 3U,
          },
      });
}

holoscan::Contract DrmBgraDisplayOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void DrmBgraDisplayOp::start() { impl_->start(); }

void DrmBgraDisplayOp::stop() { impl_->stop(); }

holoscan::expected<void, holoscan::Error> DrmBgraDisplayOp::compute(
    holoscan::ExecutionContext& context) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  const std::uint64_t frame_id = sample->metadata.frame_id;
  const std::int64_t trace_receive_us =
      holoscan::ops::FlowTrace::now_microseconds();

  const holoscan::Tensor& tensor = sample->data;
  const std::array<std::int64_t, 3U> expected_shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      4,
  };
  const std::size_t frame_bytes = frame_bytes_;
  if (tensor.data() == nullptr || tensor.device().device_type != kDLCUDA ||
      !same_dtype(tensor.dtype(), kUInt8Dtype) ||
      !shape_equals(tensor, expected_shape) || !tensor.is_contiguous() ||
      tensor.nbytes() != static_cast<std::int64_t>(frame_bytes)) {
    return failure(holoscan::ErrorCode::kInvalidArgument,
                   "DrmBgraDisplayOp expects contiguous CUDA uint8 [height,width,4] BGRA input");
  }

  try {
    impl_->wait_for_flip();
    const cudaStream_t stream = context.cuda_stream().get();
    if (stream == nullptr) {
      return failure(holoscan::ErrorCode::kNotSupported,
                     "DrmBgraDisplayOp has no runtime CUDA stream");
    }

    const std::uint32_t next_buffer = (impl_->current_buffer + 1U) % 2U;
    DrmBuffer& destination = *impl_->buffers[next_buffer];
    const std::size_t source_pitch = static_cast<std::size_t>(width_) * 4U;
    if (destination.cuda_accessible()) {
      const cudaError_t copied = cudaMemcpy2DAsync(destination.device_pointer(),
                                                   destination.pitch(),
                                                   tensor.data_as<std::uint8_t>(),
                                                   source_pitch,
                                                   source_pitch,
                                                   static_cast<std::size_t>(height_),
                                                   cudaMemcpyDefault,
                                                   stream);
      if (copied != cudaSuccess) {
        return failure(holoscan::ErrorCode::kFailure,
                       cuda_message("BGRA-to-DRM direct CUDA copy", copied));
      }

      // Scanout starts after the page flip, so all CUDA writes must be visible first.
      const cudaError_t synchronized = cudaStreamSynchronize(stream);
      if (synchronized != cudaSuccess) {
        return failure(holoscan::ErrorCode::kFailure,
                       cuda_message("cudaStreamSynchronize(DRM framebuffer)", synchronized));
      }
    } else {
      // The source may have been produced on this non-default stream. Wait for
      // it before the synchronous device-to-host copy into the DRM mapping.
      const cudaError_t synchronized = cudaStreamSynchronize(stream);
      if (synchronized != cudaSuccess) {
        return failure(holoscan::ErrorCode::kFailure,
                       cuda_message("cudaStreamSynchronize(BGRA source)", synchronized));
      }
      const cudaError_t copied = cudaMemcpy2D(destination.host_pointer(),
                                              destination.pitch(),
                                              tensor.data_as<std::uint8_t>(),
                                              source_pitch,
                                              source_pitch,
                                              static_cast<std::size_t>(height_),
                                              cudaMemcpyDeviceToHost);
      if (copied != cudaSuccess) {
        return failure(holoscan::ErrorCode::kFailure,
                       cuda_message("BGRA-to-DRM mmap copy", copied));
      }
    }

    impl_->pending_buffer = next_buffer;
    if (drmModePageFlip(impl_->drm.fd(),
                        impl_->drm.crtc_id(),
                        destination.framebuffer(),
                        DRM_MODE_PAGE_FLIP_EVENT,
                        impl_.get()) != 0) {
      impl_->pending_buffer = impl_->current_buffer;
      return failure(holoscan::ErrorCode::kFailure, errno_message("drmModePageFlip"));
    }
    impl_->flip_pending = true;
    holoscan::ops::FlowTrace::instance().record(
        frame_id,
        "display",
        trace_receive_us,
        holoscan::ops::FlowTrace::now_microseconds());
    return {};
  } catch (const std::exception& error) {
    return failure(holoscan::ErrorCode::kFailure, error.what());
  }
}

}  // namespace holoscan::ops
