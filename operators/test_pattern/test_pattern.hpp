// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
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

enum class TestPatternKind : std::uint8_t {
  kColorBars,
  kCheckerboard,
};

/// Generate contiguous CUDA BGRA Tensor frames for the Interprocess demo.
class TestPatternOp final : public holoscan::Operator<> {
 public:
  TestPatternOp(std::int32_t width, std::int32_t height, std::int32_t fps, TestPatternKind pattern);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> output;

 private:
  std::int32_t width_{};
  std::int32_t height_{};
  std::chrono::nanoseconds period_{};
  TestPatternKind pattern_{};
  std::uint64_t frame_id_{};
};

}  // namespace holoscan::ops
