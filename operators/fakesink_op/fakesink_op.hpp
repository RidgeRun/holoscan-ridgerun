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

/// Consume tensor samples without copying or inspecting their payload.
class FakesinkOp final : public holoscan::Operator<> {
 public:
  explicit FakesinkOp(bool verbose = false) : verbose_(verbose) {}

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;

 private:
  bool verbose_{};
  std::uint64_t buffers_received_{};
};

}  // namespace holoscan::ops
