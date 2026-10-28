// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::ops {

/// Fuse RG10 demosaic, RGB conversion, letterbox resize, and YOLO normalization.
class RawBayerToYoloTensorOp final : public holoscan::Operator<> {
 public:
  RawBayerToYoloTensorOp(std::int32_t source_width,
                         std::int32_t source_height,
                         std::int32_t network_width = 640,
                         std::int32_t network_height = 640,
                         std::int32_t optical_black = 0,
                         std::int32_t bayer_grid = 1,
                         std::int32_t raw_shift = 6,
                         float input_gain = 8.0F);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t source_width_;
  std::int32_t source_height_;
  std::int32_t network_width_;
  std::int32_t network_height_;
  std::int32_t resized_width_;
  std::int32_t resized_height_;
  std::int32_t padding_left_;
  std::int32_t padding_top_;
  std::int32_t optical_black_;
  std::int32_t bayer_grid_;
  std::int32_t raw_shift_;
  float input_gain_;
};

}  // namespace holoscan::ops
