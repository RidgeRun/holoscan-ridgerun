// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "v4l2_userptr_source/v4l2_userptr_source.hpp"

#include <cuda_runtime_api.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/memory/memory_storage_type.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "frame_limit/frame_limit.hpp"
#include "flow_trace/flow_trace.hpp"
#include "operator_utils/tensor_utils.hpp"

namespace holoscan::ops {
namespace {

constexpr std::uint32_t kRequestedBufferCount = 4U;
constexpr std::uint32_t kMaximumReasonableBufferCount = 64U;
constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::int64_t kNanosecondsPerMicrosecond = 1'1000LL;
constexpr std::uint32_t kTegraCameraControlBase = V4L2_CTRL_CLASS_CAMERA | 0x2000U;
constexpr std::uint32_t kTegraFrameRateControl = kTegraCameraControlBase + 11U;
constexpr std::uint32_t kTegraBypassModeControl = kTegraCameraControlBase + 100U;
constexpr std::uint32_t kTegraOverrideEnableControl = kTegraCameraControlBase + 101U;

[[nodiscard]] int retry_ioctl(int fd, unsigned long request, void* argument) noexcept {  // NOLINT
  int result = -1;
  do {
    result = ::ioctl(fd, request, argument);
  } while (result < 0 && errno == EINTR);
  return result;
}

[[nodiscard]] std::string errno_message(std::string_view operation, int error_number) {
  std::ostringstream message;
  message << operation << " failed: " << std::strerror(error_number);
  return message.str();
}

[[nodiscard]] std::string cuda_message(std::string_view operation, cudaError_t error) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(error);
  return message.str();
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> failure(holoscan::ErrorCode code,
                                                                std::string message) {
  return holoscan::make_unexpected(holoscan::Error{code, std::move(message)});
}

[[nodiscard]] std::string fourcc_string(std::uint32_t fourcc) {
  std::array<char, 5U> text{
      static_cast<char>(fourcc & 0xffU),
      static_cast<char>((fourcc >> 8U) & 0xffU),
      static_cast<char>((fourcc >> 16U) & 0xffU),
      static_cast<char>((fourcc >> 24U) & 0xffU),
      '\0',
  };
  for (std::size_t index = 0U; index < 4U; ++index) {
    if (text[index] < 0x20 || text[index] > 0x7e) {
      text[index] = '?';
    }
  }
  return std::string{text.data()};
}

}  // namespace

struct V4L2UserPtrSourceOp::CaptureState {
  std::mutex mutex;
  int fd{-1};
  bool streaming{};
};

struct V4L2UserPtrSourceOp::UserPtrBuffer {
  std::shared_ptr<CaptureState> capture;
  std::uint32_t index{};
  void* host_address{};
  void* device_address{};
  std::size_t length{};
  bool queued{};

  ~UserPtrBuffer() {
    if (host_address != nullptr) {
      const cudaError_t released = cudaFreeHost(host_address);
      if (released != cudaSuccess) {
        std::fprintf(stderr, "[v4l2_userptr_source] cudaFreeHost failed: %s\n",
                     cudaGetErrorString(released));
      }
    }
  }

  [[nodiscard]] bool requeue(int* error_number = nullptr) noexcept {
    if (capture == nullptr) {
      return true;
    }

    const std::lock_guard lock(capture->mutex);
    if (queued) {
      return true;
    }
    if (!capture->streaming || capture->fd < 0) {
      // STREAMOFF has already made this slot safe to release.
      queued = true;
      return true;
    }

    v4l2_buffer buffer{};
    buffer.index = index;
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_USERPTR;
    buffer.m.userptr = reinterpret_cast<unsigned long>(host_address);
    buffer.length = static_cast<__u32>(length);
    buffer.bytesused = 0U;
    if (retry_ioctl(capture->fd, VIDIOC_QBUF, &buffer) < 0) {
      if (error_number != nullptr) {
        *error_number = errno;
      }
      return false;
    }
    queued = true;
    return true;
  }
};

// A Tensor keeps this reference alive until the downstream consumer releases the captured slot.
// The destructor is the ownership hand-off back to V4L2; the source must not QBUF immediately
// after emitting because DRM/CUDA may still be reading that slot.
struct V4L2UserPtrSourceOp::UserPtrTensorMemoryReference final
    : public holoscan::TensorMemoryReference {
 public:
  explicit UserPtrTensorMemoryReference(std::shared_ptr<V4L2UserPtrSourceOp::UserPtrBuffer> slot)
      : slot_(std::move(slot)) {}

  // VIDIOC_DQBUF returns only after the capture device has finished writing the slot. The
  // producer is therefore synchronously ready before the Tensor is published; no CUDA event
  // needs to be inserted before a downstream CUDA consumer reads the mapped host memory.
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> synchronize_for_consumer(
      cudaStream_t) override {
    return {};
  }

  // Remember the stream that consumes this slot. The destructor waits for it before returning
  // the USERPTR buffer to V4L2, so the driver cannot overwrite a frame still being read by CUDA.
  [[nodiscard]] bool set_deallocation_stream(cudaStream_t stream) override {
    deallocation_stream_ = stream;
    has_deallocation_stream_ = true;
    return true;
  }

  ~UserPtrTensorMemoryReference() override {
    if (has_deallocation_stream_) {
      const cudaError_t status = cudaStreamSynchronize(deallocation_stream_);
      if (status != cudaSuccess) {
        std::fprintf(stderr,
                     "[v4l2_userptr_source] CUDA stream synchronization failed while releasing "
                     "USERPTR: %s\n",
                     cudaGetErrorString(status));
      }
    }
    if (slot_ != nullptr) {
      int error_number = 0;
      if (!slot_->requeue(&error_number)) {
        std::fprintf(stderr,
                     "[v4l2_userptr_source] VIDIOC_QBUF failed while releasing USERPTR: %s\n",
                     std::strerror(error_number));
      }
    }
  }

 private:
  std::shared_ptr<V4L2UserPtrSourceOp::UserPtrBuffer> slot_;
  cudaStream_t deallocation_stream_{nullptr};
  bool has_deallocation_stream_{false};
};

V4L2UserPtrSourceOp::V4L2UserPtrSourceOp(
    std::string device,
    int width,
    int height,
    int fps,
    int cuda_device,
    std::int64_t sensor_gain,
    std::int64_t exposure_us,
    holoscan::ops::FrameLimit frame_limit,
    holoscan::ops::FrameStartGate start_gate)
    : device_(std::move(device)),
      width_(width),
      height_(height),
      fps_(fps),
      cuda_device_(cuda_device),
      sensor_gain_(sensor_gain),
      exposure_us_(exposure_us),
      frame_limit_(std::move(frame_limit)),
      start_gate_(std::move(start_gate)) {
  if (device_.empty()) {
    throw std::invalid_argument("V4L2 device path must not be empty");
  }
  if (width_ <= 0 || height_ <= 0 || fps_ <= 0 || cuda_device_ < 0 || sensor_gain_ <= 0 ||
      exposure_us_ <= 0) {
    throw std::invalid_argument(
        "V4L2 width, height, fps, sensor gain, and exposure must be positive; "
        "CUDA device must be nonnegative");
  }
  if (fps_ > kNanosecondsPerSecond) {
    throw std::invalid_argument("V4L2 fps exceeds the nanosecond clock resolution");
  }
  if ((width_ & 1) != 0) {
    throw std::invalid_argument("RG10 capture width must be even");
  }

  const auto unsigned_width = static_cast<std::size_t>(width_);
  const auto unsigned_height = static_cast<std::size_t>(height_);
  if (unsigned_width > std::numeric_limits<std::size_t>::max() / 2U) {
    throw std::invalid_argument("V4L2 frame row byte count overflows size_t");
  }
  row_bytes_ = unsigned_width * 2U;
  if (unsigned_height > std::numeric_limits<std::size_t>::max() / row_bytes_) {
    throw std::invalid_argument("V4L2 frame byte count overflows size_t");
  }
  frame_bytes_ = row_bytes_ * unsigned_height;
  poll_period_ = std::chrono::nanoseconds{kNanosecondsPerSecond / fps_};
}

V4L2UserPtrSourceOp::~V4L2UserPtrSourceOp() { cleanup(); }

void V4L2UserPtrSourceOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt16Dtype,
              .rank = 3U,
          },
          .bounds = holoscan::tensor_bounds(frame_bytes_),
      });
}

holoscan::Contract V4L2UserPtrSourceOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = poll_period_});
  return result;
}

void V4L2UserPtrSourceOp::start() {
  cleanup();
  compute_seen_ = false;
  eagain_seen_ = false;
  first_frame_seen_ = false;
  capture_ = std::make_shared<CaptureState>();

  try {
    // The mapping flag must be requested before the CUDA context is created. In the normal
    // Holoscan process the runtime may already own a context, in which case CUDA reports
    // cudaErrorSetOnActiveProcess; cudaHostGetDevicePointer below remains the authoritative
    // capability check for the active context.
    const cudaError_t map_host = cudaSetDeviceFlags(cudaDeviceMapHost);
    if (map_host != cudaSuccess && map_host != cudaErrorSetOnActiveProcess) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                   cuda_message("cudaSetDeviceFlags(cudaDeviceMapHost)", map_host));
    }
    const cudaError_t selected = cudaSetDevice(cuda_device_);
    if (selected != cudaSuccess) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                   cuda_message("cudaSetDevice", selected));
    }
    cudaDeviceProp device_properties{};
    const cudaError_t queried = cudaGetDeviceProperties(&device_properties, cuda_device_);
    if (queried != cudaSuccess) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                   cuda_message("cudaGetDeviceProperties", queried));
    }
    if (device_properties.canMapHostMemory == 0) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                   "CUDA device cannot map host memory");
    }

    capture_->fd = ::open(device_.c_str(), O_RDWR | O_NONBLOCK | O_CLOEXEC);
    if (capture_->fd < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound,
                                   errno_message("open(" + device_ + ")", error_number));
    }

    v4l2_capability capability{};
    if (xioctl(VIDIOC_QUERYCAP, &capability) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   errno_message("VIDIOC_QUERYCAP", error_number));
    }
    const std::uint32_t capabilities = (capability.capabilities & V4L2_CAP_DEVICE_CAPS) != 0U
                                           ? capability.device_caps
                                           : capability.capabilities;
    if ((capabilities & V4L2_CAP_VIDEO_CAPTURE) == 0U ||
        (capabilities & V4L2_CAP_STREAMING) == 0U) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kNotSupported,
          "V4L2 device must support single-plane VIDEO_CAPTURE and STREAMING");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = static_cast<__u32>(width_);
    format.fmt.pix.height = static_cast<__u32>(height_);
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_SRGGB10;
    format.fmt.pix.field = V4L2_FIELD_NONE;
    if (xioctl(VIDIOC_S_FMT, &format) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   errno_message("VIDIOC_S_FMT", error_number));
    }
    if (format.type != V4L2_BUF_TYPE_VIDEO_CAPTURE ||
        format.fmt.pix.width != static_cast<__u32>(width_) ||
        format.fmt.pix.height != static_cast<__u32>(height_) ||
        format.fmt.pix.pixelformat != V4L2_PIX_FMT_SRGGB10 ||
        format.fmt.pix.field != V4L2_FIELD_NONE) {
      std::ostringstream message;
      message << "requested exact " << width_ << 'x' << height_
              << " progressive RG10, but the driver selected " << format.fmt.pix.width << 'x'
              << format.fmt.pix.height << ' ' << fourcc_string(format.fmt.pix.pixelformat)
              << " field=" << format.fmt.pix.field;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }

    // Argus can leave Tegra VI bypass enabled. Direct V4L2 then accepts STREAMON but never
    // completes a buffer. Apply the direct-capture state when these NVIDIA controls exist;
    // ordinary V4L2 devices simply do not expose them.
    auto set_tegra_control = [this](std::uint32_t id, std::int64_t value, const char* name) {
      v4l2_query_ext_ctrl query{};
      query.id = id;
      if (xioctl(VIDIOC_QUERY_EXT_CTRL, &query) < 0) {
        if (errno == EINVAL || errno == ENOTTY) return;
        const int error_number = errno;
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kFailure,
            errno_message(std::string{"VIDIOC_QUERY_EXT_CTRL("} + name + ")", error_number));
      }
      if ((query.flags & V4L2_CTRL_FLAG_DISABLED) != 0U) return;

      v4l2_ext_control control{};
      control.id = id;
      if (query.type == V4L2_CTRL_TYPE_INTEGER64) {
        control.value64 = value;
      } else {
        control.value = static_cast<__s32>(value);
      }
      v4l2_ext_controls controls{};
      controls.which = V4L2_CTRL_WHICH_CUR_VAL;
      controls.count = 1U;
      controls.controls = &control;
      if (xioctl(VIDIOC_S_EXT_CTRLS, &controls) < 0) {
        const int error_number = errno;
        throw holoscan::RuntimeError(
            holoscan::ErrorCode::kFailure,
            errno_message(std::string{"VIDIOC_S_EXT_CTRLS("} + name + ")", error_number));
      }
    };
    set_tegra_control(kTegraBypassModeControl, 0, "bypass_mode");
    set_tegra_control(kTegraOverrideEnableControl, 1, "override_enable");
    set_tegra_control(kTegraFrameRateControl,
                      static_cast<std::int64_t>(fps_) * 1'000'000LL,
                      "frame_rate");
    set_tegra_control(kTegraCameraControlBase + 9U, sensor_gain_, "gain");
    set_tegra_control(kTegraCameraControlBase + 10U, exposure_us_, "exposure");

    bytes_per_line_ = format.fmt.pix.bytesperline;
    if (bytes_per_line_ == 0U) {
      bytes_per_line_ = row_bytes_;
    }
    if (bytes_per_line_ != row_bytes_ ||
        static_cast<std::size_t>(height_) >
            std::numeric_limits<std::size_t>::max() / bytes_per_line_) {
      throw holoscan::RuntimeError(
          holoscan::ErrorCode::kIncompatible,
          "V4L2 USERPTR RG10 requires a contiguous width * 2 byte row stride");
    }
    const std::size_t minimum_buffer_span =
        (static_cast<std::size_t>(height_) - 1U) * bytes_per_line_ + row_bytes_;
    buffer_bytes_ = std::max<std::size_t>(minimum_buffer_span, format.fmt.pix.sizeimage);

    v4l2_streamparm stream_parameters{};
    stream_parameters.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    auto& capture_parameters = stream_parameters.parm.capture;  // codespell:ignore parm
    capture_parameters.timeperframe.numerator = 1U;
    capture_parameters.timeperframe.denominator = static_cast<__u32>(fps_);
    if (xioctl(VIDIOC_S_PARM, &stream_parameters) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   errno_message("VIDIOC_S_PARM", error_number));
    }
    const std::uint32_t numerator = capture_parameters.timeperframe.numerator;
    const std::uint32_t denominator = capture_parameters.timeperframe.denominator;
    if (numerator == 0U || denominator == 0U ||
        static_cast<std::uint64_t>(denominator) != static_cast<std::uint64_t>(fps_) * numerator) {
      std::ostringstream message;
      message << "requested exactly " << fps_ << " fps, but the driver selected " << numerator
              << '/' << denominator << " seconds per frame";
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }

    v4l2_requestbuffers request{};
    request.count = kRequestedBufferCount;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_USERPTR;
    if (xioctl(VIDIOC_REQBUFS, &request) < 0) {
      const int error_number = errno;
      const auto code = error_number == EINVAL ? holoscan::ErrorCode::kNotSupported
                                               : holoscan::ErrorCode::kFailure;
      throw holoscan::RuntimeError(code, errno_message("VIDIOC_REQBUFS(USERPTR)", error_number));
    }
    if (request.count == 0U || request.count > kMaximumReasonableBufferCount) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kResourceExhausted,
                                   "V4L2 returned an invalid USERPTR buffer count");
    }

    buffers_.reserve(request.count);
    for (std::uint32_t index = 0U; index < request.count; ++index) {
      auto slot = std::make_shared<UserPtrBuffer>();
      slot->capture = capture_;
      slot->index = index;
      slot->length = buffer_bytes_;

      void* host_address = nullptr;
      const cudaError_t allocated =
          cudaHostAlloc(&host_address, slot->length, cudaHostAllocMapped | cudaHostAllocPortable);
      if (allocated != cudaSuccess) {
        throw holoscan::RuntimeError(holoscan::ErrorCode::kResourceExhausted,
                                     cuda_message("cudaHostAlloc(USERPTR)", allocated));
      }
      slot->host_address = host_address;

      void* device_address = nullptr;
      const cudaError_t mapped = cudaHostGetDevicePointer(&device_address, host_address, 0U);
      if (mapped != cudaSuccess || device_address == nullptr) {
        throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                     cuda_message("cudaHostGetDevicePointer(USERPTR)", mapped));
      }
      slot->device_address = device_address;

      v4l2_buffer buffer{};
      buffer.index = index;
      buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      buffer.memory = V4L2_MEMORY_USERPTR;
      buffer.m.userptr = reinterpret_cast<unsigned long>(slot->host_address);
      buffer.length = static_cast<__u32>(slot->length);
      buffer.bytesused = 0U;
      if (xioctl(VIDIOC_QBUF, &buffer) < 0) {
        const int error_number = errno;
        const auto code = error_number == EINVAL ? holoscan::ErrorCode::kNotSupported
                                               : holoscan::ErrorCode::kFailure;
      throw holoscan::RuntimeError(
          code, errno_message("VIDIOC_QBUF(USERPTR): driver rejected CUDA-mapped buffer",
                              error_number));
      }
      slot->queued = true;
      buffers_.push_back(std::move(slot));
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(VIDIOC_STREAMON, &type) < 0) {
      const int error_number = errno;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   errno_message("VIDIOC_STREAMON", error_number));
    }
    capture_->streaming = true;

    std::fprintf(stdout,
                 "[v4l2_userptr_source] streaming %dx%d RG10 at %d fps from %s "
                 "(stride=%zu, buffers=%zu, gain=%lld, exposure=%lld us, CUDA-mapped USERPTR)\n",
                 width_, height_, fps_, device_.c_str(), bytes_per_line_, buffers_.size(),
                 static_cast<long long>(sensor_gain_), static_cast<long long>(exposure_us_));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "[v4l2_userptr_source] startup failed for %s: %s\n", device_.c_str(),
                 error.what());
    cleanup();
    throw;
  } catch (...) {
    std::fprintf(stderr, "[v4l2_userptr_source] startup failed for %s: unknown error\n",
                 device_.c_str());
    cleanup();
    throw;
  }
}

void V4L2UserPtrSourceOp::stop() { cleanup(); }

holoscan::expected<void, holoscan::Error> V4L2UserPtrSourceOp::compute(
    holoscan::ExecutionContext& context) {
  if (frame_limit_.reached()) {
    return {};
  }
  if (!compute_seen_) {
    std::fprintf(stderr,
                 "[v4l2_userptr_source] compute activated; waiting for first RG10 frame\n");
    compute_seen_ = true;
  }
  if (capture_ == nullptr || capture_->fd < 0 || !capture_->streaming) {
    return failure(holoscan::ErrorCode::kNotReady, "V4L2 USERPTR source is not streaming");
  }

  v4l2_buffer buffer{};
  buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buffer.memory = V4L2_MEMORY_USERPTR;
  if (xioctl(VIDIOC_DQBUF, &buffer) < 0) {
    const int error_number = errno;
    if (error_number == EAGAIN || error_number == EWOULDBLOCK) {
      if (!eagain_seen_) {
        std::fprintf(stderr, "[v4l2_userptr_source] first dequeue returned EAGAIN\n");
        eagain_seen_ = true;
      }
      return {};
    }
    return failure(holoscan::ErrorCode::kFailure,
                   errno_message("VIDIOC_DQBUF(USERPTR)", error_number));
  }
  const std::int64_t trace_receive_us = FlowTrace::now_microseconds();

  const auto host_address = reinterpret_cast<void*>(static_cast<std::uintptr_t>(buffer.m.userptr));
  const auto slot_iterator =
      std::find_if(buffers_.begin(), buffers_.end(), [host_address](const auto& candidate) {
        return candidate != nullptr && candidate->host_address == host_address;
      });
  if (slot_iterator == buffers_.end()) {
    return failure(holoscan::ErrorCode::kProtocolError,
                   "V4L2 returned a USERPTR address that was not allocated by this source");
  }
  const auto& slot = *slot_iterator;
  slot->queued = false;
  if (!first_frame_seen_) {
    std::fprintf(stdout,
                 "[v4l2_userptr_source] first RG10 frame dequeued: index=%u bytesused=%u\n",
                 buffer.index,
                 buffer.bytesused);
    first_frame_seen_ = true;
  }

  auto requeue_now = [&slot](int* error_number =
                                 nullptr) -> holoscan::expected<void, holoscan::Error> {
    if (slot->requeue(error_number)) {
      return {};
    }
    const int code = error_number != nullptr ? *error_number : errno;
    return failure(holoscan::ErrorCode::kFailure, errno_message("VIDIOC_QBUF(USERPTR)", code));
  };

  if (!start_gate_.is_open()) {
    return requeue_now();
  }

  if ((buffer.flags & V4L2_BUF_FLAG_ERROR) != 0U) {
    return requeue_now();
  }

  const std::size_t required_source_span =
      (static_cast<std::size_t>(height_) - 1U) * bytes_per_line_ + row_bytes_;
  if (slot->host_address == nullptr || slot->device_address == nullptr ||
      slot->length < required_source_span || buffer.bytesused < required_source_span ||
      static_cast<std::size_t>(buffer.bytesused) > slot->length) {
    int error_number = 0;
    auto requeued = requeue_now(&error_number);
    if (!requeued) {
      return requeued;
    }
    return failure(holoscan::ErrorCode::kProtocolError,
                   "V4L2 USERPTR frame payload does not cover the negotiated row-strided image");
  }

  const std::array<std::int64_t, 3U> shape{
      static_cast<std::int64_t>(height_),
      static_cast<std::int64_t>(width_),
      1,
  };
  auto memory_reference = std::make_shared<UserPtrTensorMemoryReference>(slot);
  auto tensor =
      holoscan::make_tensor(shape, kUInt16Dtype, std::nullopt, holoscan::MemoryStorageType::kDevice,
                            slot->device_address, std::move(memory_reference), cuda_device_);
  if (!tensor) {
    // The reference's destructor returns the slot to V4L2.
    return holoscan::make_unexpected(std::move(tensor).error());
  }

  const holoscan::TimePoint captured = capture_time(buffer, context.activation_time());
  const std::uint64_t current_frame_id = ++frame_id_;
  auto emitted = frame.emit(
      std::move(*tensor),
      holoscan::EmitOptions{.capture_time = captured, .frame_id = current_frame_id});
  if (!emitted) {
    return emitted;
  }
  frame_limit_.record_frame();
  FlowTrace::instance().record(current_frame_id,
                                    "camera",
                                    trace_receive_us,
                                    FlowTrace::now_microseconds());
  return {};
}

int V4L2UserPtrSourceOp::xioctl(unsigned long request,  // NOLINT(runtime/int)
                                void* argument) const noexcept {
  return capture_ == nullptr ? -1 : retry_ioctl(capture_->fd, request, argument);
}

void V4L2UserPtrSourceOp::cleanup() noexcept {
  auto capture = std::move(capture_);
  if (capture != nullptr) {
    const std::lock_guard lock(capture->mutex);
    if (capture->streaming) {
      v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
      if (retry_ioctl(capture->fd, VIDIOC_STREAMOFF, &type) < 0) {
        std::fprintf(stderr, "[v4l2_userptr_source] VIDIOC_STREAMOFF failed during cleanup: %s\n",
                     std::strerror(errno));
      }
      capture->streaming = false;
    }
    if (capture->fd >= 0) {
      if (::close(capture->fd) < 0) {
        std::fprintf(stderr, "[v4l2_userptr_source] close failed during cleanup: %s\n",
                     std::strerror(errno));
      }
      capture->fd = -1;
    }
  }

  // Slots retained by emitted tensors survive this clear. Their memory-reference destructor sees
  // the inactive CaptureState and releases CUDA host memory without attempting another QBUF.
  buffers_.clear();
  bytes_per_line_ = 0U;
  buffer_bytes_ = 0U;
}

holoscan::TimePoint V4L2UserPtrSourceOp::capture_time(
    const v4l2_buffer& buffer, holoscan::TimePoint activation_time) const noexcept {
  if (!activation_time.valid()) {
    return {};
  }

  if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) {
    timespec now{};
    if (::clock_gettime(CLOCK_MONOTONIC, &now) == 0 && buffer.timestamp.tv_sec >= 0 &&
        buffer.timestamp.tv_usec >= 0 && buffer.timestamp.tv_usec < 1'000'000) {
      const std::int64_t now_ns =
          static_cast<std::int64_t>(now.tv_sec) * kNanosecondsPerSecond + now.tv_nsec;
      const std::int64_t captured_ns =
          static_cast<std::int64_t>(buffer.timestamp.tv_sec) * kNanosecondsPerSecond +
          static_cast<std::int64_t>(buffer.timestamp.tv_usec) * kNanosecondsPerMicrosecond;
      if (captured_ns <= now_ns) {
        const std::int64_t age_ns = now_ns - captured_ns;
        return holoscan::TimePoint{
            .timestamp_ns = activation_time.timestamp_ns - age_ns,
            .clock_id = activation_time.clock_id,
        };
      }
    }
  }

  return activation_time;
}

}  // namespace holoscan::ops
