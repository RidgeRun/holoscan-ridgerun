// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "argus_camera/argus_camera.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "frame_limit/frame_limit.hpp"
#include "operator_utils/tensor_utils.hpp"
#include "flow_trace/flow_trace.hpp"

namespace holoscan::ops {
namespace {

using holoscan::ops::kUInt8Dtype;

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;

[[nodiscard]] std::string argus_message(const char* operation, Argus::Status status) {
  std::ostringstream message;
  message << operation << " failed (Argus status " << static_cast<int>(status) << ')';
  return message.str();
}

[[nodiscard]] std::string cuda_driver_message(const char* operation, CUresult status) {
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(cuGetErrorName(status, &name));
  static_cast<void>(cuGetErrorString(status, &description));
  std::ostringstream message;
  message << operation << " failed";
  if (name != nullptr) {
    message << " (" << name << ')';
  }
  if (description != nullptr) {
    message << ": " << description;
  }
  return message.str();
}

void require_cuda(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 cuda_driver_message(operation, status));
  }
}

void require_argus(Argus::Status status, const char* operation) {
  if (status != Argus::STATUS_OK) {
    throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                 argus_message(operation, status));
  }
}

class ScopedCudaContext final {
 public:
  explicit ScopedCudaContext(CUcontext context) {
    require_cuda(cuCtxPushCurrent(context), "cuCtxPushCurrent");
    active_ = true;
  }

  ScopedCudaContext(const ScopedCudaContext&) = delete;
  ScopedCudaContext& operator=(const ScopedCudaContext&) = delete;

  ~ScopedCudaContext() {
    if (active_) {
      CUcontext popped = nullptr;
      static_cast<void>(cuCtxPopCurrent(&popped));
    }
  }

 private:
  bool active_{};
};

class AcquiredFrame final {
 public:
  AcquiredFrame(CUeglStreamConnection* connection, CUgraphicsResource resource, CUstream stream)
      : connection_(connection), resource_(resource), stream_(stream) {}
  AcquiredFrame(const AcquiredFrame&) = delete;
  AcquiredFrame& operator=(const AcquiredFrame&) = delete;

  ~AcquiredFrame() {
    if (resource_ != nullptr) {
      static_cast<void>(cuEGLStreamConsumerReleaseFrame(
          connection_, resource_, &stream_));
    }
  }

  [[nodiscard]] CUresult release() {
    if (resource_ == nullptr) {
      return CUDA_SUCCESS;
    }
    const CUresult result =
        cuEGLStreamConsumerReleaseFrame(connection_, resource_, &stream_);
    if (result == CUDA_SUCCESS) {
      resource_ = nullptr;
    }
    return result;
  }

 private:
  CUeglStreamConnection* connection_{};
  CUgraphicsResource resource_{};
  CUstream stream_{};
};

__device__ __forceinline__ std::uint8_t clamp_byte(int value) {
  value = value < 0 ? 0 : value;
  return static_cast<std::uint8_t>(value > 255 ? 255 : value);
}

__global__ void nv12_to_bgra_kernel(cudaSurfaceObject_t y_plane,
                                    cudaSurfaceObject_t uv_plane,
                                    std::uint8_t* output,
                                    std::int32_t width,
                                    std::int32_t height) {
  const std::int32_t x = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  const std::int32_t y = static_cast<std::int32_t>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= width || y >= height) {
    return;
  }

  const auto luma = static_cast<int>(surf2Dread<std::uint8_t>(y_plane, x, y));
  const uchar2 chroma = surf2Dread<uchar2>(
      uv_plane, (x / 2) * static_cast<int>(sizeof(uchar2)), y / 2);
  const int c = luma > 16 ? luma - 16 : 0;
  const int u = static_cast<int>(chroma.x) - 128;
  const int v = static_cast<int>(chroma.y) - 128;

  // Limited-range BT.709, matching Argus preview NV12 output.
  const int red = (298 * c + 459 * v + 128) >> 8;
  const int green = (298 * c - 55 * u - 136 * v + 128) >> 8;
  const int blue = (298 * c + 541 * u + 128) >> 8;
  const std::size_t offset =
      (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
       static_cast<std::size_t>(x)) * 4U;
  output[offset] = clamp_byte(blue);
  output[offset + 1U] = clamp_byte(green);
  output[offset + 2U] = clamp_byte(red);
  output[offset + 3U] = 255U;
}

}  // namespace

ArgusCameraOp::ArgusCameraOp(std::uint32_t camera_index,
                             std::uint32_t sensor_mode_index,
                             std::int32_t width,
                             std::int32_t height,
                             std::int32_t fps,
                             holoscan::ops::FrameLimit frame_limit)
    : camera_index_(camera_index),
      sensor_mode_index_(sensor_mode_index),
      width_(width),
      height_(height),
      fps_(fps),
      frame_limit_(std::move(frame_limit)) {
  if (width_ <= 0 || height_ <= 0 || fps_ <= 0) {
    throw std::invalid_argument("Argus width, height, and fps must be positive");
  }
  if ((width_ & 1) != 0 || (height_ & 1) != 0) {
    throw std::invalid_argument("Argus NV12 width and height must be even");
  }
  const auto pixels =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  if (pixels > std::numeric_limits<std::size_t>::max() / 4U) {
    throw std::overflow_error("Argus BGRA frame size overflows size_t");
  }
  frame_bytes_ = pixels * 4U;
  poll_period_ = std::chrono::nanoseconds{kNanosecondsPerSecond / fps_};
}

ArgusCameraOp::~ArgusCameraOp() { cleanup(); }

void ArgusCameraOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = kUInt8Dtype,
              .rank = 3U,
          },
          .bounds = holoscan::tensor_bounds(frame_bytes_),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract ArgusCameraOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = poll_period_});
  return result;
}

void ArgusCameraOp::start() {
  cleanup();
  try {
    Argus::Status status = Argus::STATUS_OK;
    camera_provider_.reset(Argus::CameraProvider::create(&status));
    require_argus(status, "CameraProvider::create");
    if (!camera_provider_) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotReady,
                                   "Argus returned a null camera provider");
    }

    auto* provider = Argus::interface_cast<Argus::ICameraProvider>(camera_provider_);
    if (provider == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus ICameraProvider is unavailable");
    }
    std::vector<Argus::CameraDevice*> devices;
    require_argus(provider->getCameraDevices(&devices), "getCameraDevices");
    if (camera_index_ >= devices.size()) {
      std::ostringstream message;
      message << "Argus camera index " << camera_index_ << " is unavailable; found "
              << devices.size() << " camera(s)";
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound, message.str());
    }
    Argus::CameraDevice* const device = devices[camera_index_];
    auto* properties = Argus::interface_cast<Argus::ICameraProperties>(device);
    if (properties == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus ICameraProperties is unavailable");
    }
    std::vector<Argus::SensorMode*> modes;
    require_argus(properties->getAllSensorModes(&modes), "getAllSensorModes");
    if (sensor_mode_index_ >= modes.size()) {
      std::ostringstream message;
      message << "Argus sensor mode " << sensor_mode_index_ << " is unavailable; camera "
              << camera_index_ << " has " << modes.size() << " mode(s)";
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotFound, message.str());
    }
    Argus::SensorMode* const mode = modes[sensor_mode_index_];
    auto* mode_interface = Argus::interface_cast<Argus::ISensorMode>(mode);
    if (mode_interface == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus ISensorMode is unavailable");
    }
    const Argus::Size2D<std::uint32_t> resolution = mode_interface->getResolution();
    if (resolution.width() != static_cast<std::uint32_t>(width_) ||
        resolution.height() != static_cast<std::uint32_t>(height_)) {
      std::ostringstream message;
      message << "sensor mode " << sensor_mode_index_ << " is " << resolution.width() << 'x'
              << resolution.height() << ", but --width/--height requested " << width_ << 'x'
              << height_;
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }
    const std::uint64_t frame_duration =
        static_cast<std::uint64_t>(kNanosecondsPerSecond / fps_);
    const Argus::Range<std::uint64_t> duration_range = mode_interface->getFrameDurationRange();
    if (frame_duration < duration_range.min() || frame_duration > duration_range.max()) {
      std::ostringstream message;
      message << fps_ << " fps is outside sensor mode " << sensor_mode_index_
              << " frame-duration range [" << duration_range.min() << ", "
              << duration_range.max() << "] ns";
      throw holoscan::RuntimeError(holoscan::ErrorCode::kIncompatible, message.str());
    }

    capture_session_.reset(provider->createCaptureSession(device, &status));
    require_argus(status, "createCaptureSession");
    capture_session_interface_ =
        Argus::interface_cast<Argus::ICaptureSession>(capture_session_);
    if (capture_session_interface_ == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus ICaptureSession is unavailable");
    }

    stream_settings_.reset(capture_session_interface_->createOutputStreamSettings(
        Argus::STREAM_TYPE_EGL, &status));
    require_argus(status, "createOutputStreamSettings");
    auto* settings = Argus::interface_cast<Argus::IOutputStreamSettings>(stream_settings_);
    auto* egl_settings =
        Argus::interface_cast<Argus::IEGLOutputStreamSettings>(stream_settings_);
    if (settings == nullptr || egl_settings == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus EGL output stream settings are unavailable");
    }
    require_argus(settings->setCameraDevice(device), "setCameraDevice");
    require_argus(egl_settings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888),
                  "setPixelFormat");
    require_argus(egl_settings->setResolution(resolution), "setResolution");
    require_argus(egl_settings->setMode(Argus::EGL_STREAM_MODE_MAILBOX), "setMode");

    output_stream_.reset(
        capture_session_interface_->createOutputStream(stream_settings_.get(), &status));
    require_argus(status, "createOutputStream");
    egl_output_stream_ = Argus::interface_cast<Argus::IEGLOutputStream>(output_stream_);
    if (egl_output_stream_ == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus IEGLOutputStream is unavailable");
    }

    require_cuda(cuInit(0), "cuInit");
    require_cuda(cuDeviceGet(&cuda_device_, 0), "cuDeviceGet");
    int integrated = 0;
    require_cuda(cuDeviceGetAttribute(&integrated,
                                      CU_DEVICE_ATTRIBUTE_INTEGRATED,
                                      cuda_device_),
                 "cuDeviceGetAttribute(INTEGRATED)");
    if (integrated == 0) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kNotSupported,
                                   "ArgusCameraOp requires the Jetson integrated GPU");
    }
    require_cuda(cuDevicePrimaryCtxRetain(&cuda_context_, cuda_device_),
                 "cuDevicePrimaryCtxRetain");
    {
      ScopedCudaContext current{cuda_context_};
      require_cuda(cuEGLStreamConsumerConnect(
                       &cuda_connection_, egl_output_stream_->getEGLStream()),
                   "cuEGLStreamConsumerConnect");
    }

    request_.reset(capture_session_interface_->createRequest(
        Argus::CAPTURE_INTENT_PREVIEW, &status));
    require_argus(status, "createRequest");
    auto* request = Argus::interface_cast<Argus::IRequest>(request_);
    auto* source = Argus::interface_cast<Argus::ISourceSettings>(request_);
    if (request == nullptr || source == nullptr) {
      throw holoscan::RuntimeError(holoscan::ErrorCode::kFailure,
                                   "Argus request/source settings are unavailable");
    }
    require_argus(source->setSensorMode(mode), "setSensorMode");
    require_argus(source->setFrameDurationRange(Argus::Range<std::uint64_t>(frame_duration)),
                  "setFrameDurationRange");
    require_argus(request->enableOutputStream(output_stream_.get()), "enableOutputStream");
    require_argus(capture_session_interface_->repeat(request_.get()), "repeat capture request");
    repeating_ = true;
    require_argus(egl_output_stream_->waitUntilConnected(), "waitUntilConnected");

    std::fprintf(stdout,
                 "[argus_camera] camera=%u mode=%u %dx%d at %d fps, "
                 "Argus NV12 EGLStream -> CUDA BGRA\n",
                 camera_index_,
                 sensor_mode_index_,
                 width_,
                 height_,
                 fps_);
  } catch (...) {
    cleanup();
    throw;
  }
}

void ArgusCameraOp::stop() { cleanup(); }

cudaSurfaceObject_t ArgusCameraOp::surface_for(CUarray array) {
  if (const auto found = surfaces_.find(array); found != surfaces_.end()) {
    return found->second;
  }
  cudaResourceDesc descriptor{};
  descriptor.resType = cudaResourceTypeArray;
  descriptor.res.array.array = reinterpret_cast<cudaArray_t>(array);
  cudaSurfaceObject_t surface{};
  const cudaError_t status = cudaCreateSurfaceObject(&surface, &descriptor);
  if (status != cudaSuccess) {
    throw holoscan::RuntimeError(
        holoscan::ErrorCode::kFailure,
        std::string("cudaCreateSurfaceObject failed: ") + cudaGetErrorString(status));
  }
  surfaces_.emplace(array, surface);
  return surface;
}

holoscan::expected<void, holoscan::Error> ArgusCameraOp::compute(
    holoscan::ExecutionContext& context) {
  try {
    if (frame_limit_.reached()) {
      return {};
    }
    const std::int64_t trace_receive_us = holoscan::ops::FlowTrace::now_microseconds();
    if (cuda_context_ == nullptr || cuda_connection_ == nullptr) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kNotReady, "Argus camera is not started"});
    }
    ScopedCudaContext current{cuda_context_};
    cudaStream_t runtime_stream = context.cuda_stream().get();
    if (runtime_stream == nullptr) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kNotSupported, "CUDA stream is unavailable"});
    }
    CUstream stream = reinterpret_cast<CUstream>(runtime_stream);
    CUgraphicsResource resource = nullptr;
    const CUresult acquired = cuEGLStreamConsumerAcquireFrame(
        &cuda_connection_, &resource, &stream, CUDA_EGL_INFINITE_TIMEOUT);
    if (acquired != CUDA_SUCCESS) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          cuda_driver_message("cuEGLStreamConsumerAcquireFrame", acquired)});
    }
    AcquiredFrame acquired_frame{&cuda_connection_, resource, stream};

    CUeglFrame egl_frame{};
    const CUresult mapped = cuGraphicsResourceGetMappedEglFrame(
        &egl_frame, resource, 0U, 0U);
    if (mapped != CUDA_SUCCESS) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          cuda_driver_message("cuGraphicsResourceGetMappedEglFrame", mapped)});
    }
    if (egl_frame.frameType != CU_EGL_FRAME_TYPE_ARRAY || egl_frame.planeCount < 2U ||
        egl_frame.frame.pArray[0] == nullptr || egl_frame.frame.pArray[1] == nullptr ||
        egl_frame.width != static_cast<unsigned int>(width_) ||
        egl_frame.height != static_cast<unsigned int>(height_)) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kIncompatible,
          "Argus EGLStream did not provide the expected two-array NV12 frame"});
    }

    const std::array<std::int64_t, 3U> shape{
        static_cast<std::int64_t>(height_),
        static_cast<std::int64_t>(width_),
        4,
    };
    auto loan = frame.allocate_tensor(
        holoscan::TensorLoanRequest{.shape = shape, .dtype = kUInt8Dtype});
    if (!loan) {
      return holoscan::make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return holoscan::make_unexpected(std::move(writer).error());
    }
    if (writer->data() == nullptr || writer->byte_size() != frame_bytes_) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure, "invalid Argus BGRA output allocation"});
    }

    const cudaSurfaceObject_t y_surface = surface_for(egl_frame.frame.pArray[0]);
    const cudaSurfaceObject_t uv_surface = surface_for(egl_frame.frame.pArray[1]);
    constexpr dim3 block{16U, 16U, 1U};
    const dim3 grid{
        static_cast<unsigned int>((width_ + static_cast<int>(block.x) - 1) /
                                  static_cast<int>(block.x)),
        static_cast<unsigned int>((height_ + static_cast<int>(block.y) - 1) /
                                  static_cast<int>(block.y)),
        1U,
    };
    nv12_to_bgra_kernel<<<grid, block, 0U, runtime_stream>>>(
        y_surface, uv_surface, writer->data_as<std::uint8_t>(), width_, height_);
    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          std::string("NV12-to-BGRA kernel launch failed: ") + cudaGetErrorString(status)});
    }
    const CUresult released = acquired_frame.release();
    if (released != CUDA_SUCCESS) {
      return holoscan::make_unexpected(holoscan::Error{
          holoscan::ErrorCode::kFailure,
          cuda_driver_message("cuEGLStreamConsumerReleaseFrame", released)});
    }

    auto committed = std::move(*writer).commit();
    if (!committed) {
      return holoscan::make_unexpected(std::move(committed).error());
    }
    const std::uint64_t current_frame_id = ++frame_id_;
    auto emitted = frame.emit(
        std::move(*loan), holoscan::EmitOptions{.frame_id = current_frame_id});
    if (!emitted) {
      return emitted;
    }
    frame_limit_.record_frame();
    holoscan::ops::FlowTrace::instance().record(
        current_frame_id, "camera", trace_receive_us, holoscan::ops::FlowTrace::now_microseconds());
    return {};
  } catch (const holoscan::RuntimeError& error) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
  } catch (const std::exception& error) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kFailure, error.what()});
  }
}

void ArgusCameraOp::cleanup() noexcept {
  bool context_current = false;
  if (cuda_context_ != nullptr && cuCtxPushCurrent(cuda_context_) == CUDA_SUCCESS) {
    context_current = true;
  }

  if (repeating_ && capture_session_interface_ != nullptr) {
    static_cast<void>(capture_session_interface_->stopRepeat());;
    static_cast<void>(capture_session_interface_->waitForIdle());
  }
  repeating_ = false;

  if (context_current) {
    static_cast<void>(cuCtxSynchronize());
    for (const auto& [array, surface] : surfaces_) {
      static_cast<void>(array);
      static_cast<void>(cudaDestroySurfaceObject(surface));
    }
    if (cuda_connection_ != nullptr) {
      static_cast<void>(cuEGLStreamConsumerDisconnect(&cuda_connection_));
    }
  }
  surfaces_.clear();
  cuda_connection_ = nullptr;

  request_.reset();
  output_stream_.reset();
  stream_settings_.reset();
  capture_session_.reset();
  camera_provider_.reset();
  capture_session_interface_ = nullptr;
  egl_output_stream_ = nullptr;

  if (context_current) {
    CUcontext popped = nullptr;
    static_cast<void>(cuCtxPopCurrent(&popped));
  }
  if (cuda_context_ != nullptr) {
    static_cast<void>(cuDevicePrimaryCtxRelease(cuda_device_));
  }
  cuda_context_ = nullptr;
}

}  // namespace holoscan::ops
