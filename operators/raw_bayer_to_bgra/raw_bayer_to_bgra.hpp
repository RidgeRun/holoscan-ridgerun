// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include <cuda_runtime_api.h>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::ops {

class ExportableHoloMQPool;

/**
 * @brief Correct and demosaic an RG10 Bayer CUDA tensor into BGRA.
 *
 * The operator follows the processing stages used by the SDK v4l2_imx274
 * example: optical-black subtraction, Bayer demosaicing, digital display gain,
 * and gamma encoding. White balance can use the default per-frame gray-world
 * estimate or fixed R/G/B gains. Input is uint16 RG10 [height,width,1]; output
 * is uint8 BGRA [height,width,4].
 */
class RawBayerToBgraOp final : public holoscan::Operator<> {
 public:
  RawBayerToBgraOp(std::int32_t width,
                   std::int32_t height,
                   std::int32_t optical_black = 100,
                   std::int32_t bayer_grid = 1,
                   std::int32_t cuda_device = 0,
                   std::int32_t raw_shift = 0,
                   float display_gain = 1.0F,
                   float gamma = 1.0F,
                   std::optional<std::array<float, 3U>> fixed_white_balance = std::nullopt,
                   bool require_exportable_output = false);
  ~RawBayerToBgraOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  void cleanup() noexcept;

  std::int32_t width_;
  std::int32_t height_;
  std::int32_t optical_black_;
  std::int32_t bayer_grid_;
  std::int32_t cuda_device_;
  std::int32_t raw_shift_;
  float display_gain_;
  float gamma_;
  std::optional<std::array<float, 3U>> fixed_white_balance_;
  bool require_exportable_output_{};
  std::size_t input_bytes_{};
  std::size_t output_bytes_{};
  std::uint64_t red_count_{};
  std::uint64_t green_count_{};
  std::uint64_t blue_count_{};
  unsigned long long* channel_sums_{};
  float* channel_gains_{};
  std::unique_ptr<ExportableHoloMQPool> holomq_output_pool_;
  cudaStream_t holomq_output_stream_{};
  cudaStream_t owned_output_stream_{};
  bool first_frame_logged_{};
};

}  // namespace holoscan::ops
