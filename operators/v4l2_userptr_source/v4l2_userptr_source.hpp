// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/time_point.hpp>

#include "frame_limit/frame_limit.hpp"

#include <memory>
#include <string>
#include <vector>

struct v4l2_buffer;

namespace holoscan::ops {

/**
 * @brief V4L2 source using application-owned CUDA-mapped USERPTR buffers.
 *
 * Each capture slot is allocated with cudaHostAlloc(cudaHostAllocMapped), queued to
 * V4L2 as V4L2_MEMORY_USERPTR, and exposed to CUDA through cudaHostGetDevicePointer().
 * Therefore the camera writes the same backing memory that downstream CUDA operators read;
 * no host-to-device memcpy is performed by this operator.
 *
 * The backing allocation is page-locked host/system memory, not CUDA VRAM. Passing a pointer
 * returned by cudaMalloc() to a regular V4L2 USERPTR capture device is not a valid substitute:
 * USERPTR drivers pin user virtual pages for DMA. On Tegra, CUDA-mapped pinned memory provides both the user pages required by V4L2
 * and a CUDA-visible alias; USERPTR support still depends on the capture driver.
 */
class V4L2UserPtrSourceOp final : public holoscan::Operator<> {
 public:
  /**
   * @param device V4L2 device path, for example `/dev/video0`.
   * @param width Required capture width in pixels.
   * @param height Required capture height in pixels.
   * @param fps Required capture rate.
   * @param cuda_device CUDA device used to obtain the mapped device aliases.
   */
  V4L2UserPtrSourceOp(
      std::string device,
      int width,
      int height,
      int fps = 30,
      int cuda_device = 0,
      std::int64_t sensor_gain = 8'000'000,
      std::int64_t exposure_us = 15'649,
      holoscan::ops::FrameLimit frame_limit = holoscan::ops::FrameLimit{},
      holoscan::ops::FrameStartGate start_gate = holoscan::ops::FrameStartGate{});
  ~V4L2UserPtrSourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// CUDA-visible uint16 tensor with logical shape `[height, width, 1]`.
  holoscan::Output<holoscan::Tensor> frame;

 private:
  struct CaptureState;
  struct UserPtrBuffer;
  struct UserPtrTensorMemoryReference;

  [[nodiscard]] int xioctl(unsigned long request,  // NOLINT(runtime/int)
                           void* argument) const noexcept;
  void cleanup() noexcept;
  [[nodiscard]] holoscan::TimePoint capture_time(
      const struct v4l2_buffer& buffer, holoscan::TimePoint activation_time) const noexcept;

  std::string device_;
  int width_{};
  int height_{};
  int fps_{};
  int cuda_device_{};
  std::int64_t sensor_gain_{};
  std::int64_t exposure_us_{};
  std::chrono::nanoseconds poll_period_{};
  std::size_t row_bytes_{};
  std::size_t bytes_per_line_{};
  std::size_t frame_bytes_{};
  std::size_t buffer_bytes_{};
  holoscan::ops::FrameLimit frame_limit_;
  holoscan::ops::FrameStartGate start_gate_;
  std::shared_ptr<CaptureState> capture_;
  std::vector<std::shared_ptr<UserPtrBuffer>> buffers_;
  std::uint64_t frame_id_{};
  bool compute_seen_{};
  bool eagain_seen_{};
  bool first_frame_seen_{};
};

}  // namespace holoscan::ops
