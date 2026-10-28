// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_runtime_api.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <unordered_map>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

#include "frame_limit/frame_limit.hpp"

namespace holoscan::ops {

/**
 * Capture a directly attached Jetson CSI camera through libArgus.
 *
 * Argus produces NV12 into an EGLStream. CUDA consumes the stream as two
 * arrays, and one kernel reads those arrays directly while writing the
 * plan-owned BGRA output tensor. There is no staging NV12 allocation or
 * camera-to-CUDA memcpy.
 */
class ArgusCameraOp final : public holoscan::Operator<> {
 public:
  ArgusCameraOp(std::uint32_t camera_index,
                std::uint32_t sensor_mode_index,
                std::int32_t width,
                std::int32_t height,
                std::int32_t fps,
                holoscan::ops::FrameLimit frame_limit = holoscan::ops::FrameLimit{});
  ~ArgusCameraOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// Contiguous CUDA uint8 tensor with shape `[height, width, 4]` in BGRA order.
  holoscan::Output<holoscan::Tensor> frame;

 private:
  [[nodiscard]] cudaSurfaceObject_t surface_for(CUarray array);
  void cleanup() noexcept;

  std::uint32_t camera_index_{};
  std::uint32_t sensor_mode_index_{};
  std::int32_t width_{};
  std::int32_t height_{};
  std::int32_t fps_{};
  std::size_t frame_bytes_{};
  std::chrono::nanoseconds poll_period_{};
  holoscan::ops::FrameLimit frame_limit_;
  std::uint64_t frame_id_{};

  Argus::UniqueObj<Argus::CameraProvider> camera_provider_;
  Argus::UniqueObj<Argus::CaptureSession> capture_session_;
  Argus::UniqueObj<Argus::OutputStreamSettings> stream_settings_;
  Argus::UniqueObj<Argus::OutputStream> output_stream_;
  Argus::UniqueObj<Argus::Request> request_;
  Argus::ICaptureSession* capture_session_interface_{};
  Argus::IEGLOutputStream* egl_output_stream_{};

  CUdevice cuda_device_{};
  CUcontext cuda_context_{};
  CUeglStreamConnection cuda_connection_{};
  std::unordered_map<CUarray, cudaSurfaceObject_t> surfaces_;
  bool repeating_{};
};

}  // namespace holoscan::ops
