// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::ops {

/**
 * @brief Convert CUDA YUYV tensors directly into a DRM/KMS framebuffer.
 *
 * This sink does not use X11 or Wayland. It requires the process to be DRM
 * master, which normally means that the display manager must be stopped. DRM
 * framebuffers are exported as DMABUFs and imported into CUDA, so the CUDA
 * conversion writes directly into the scanout buffers without host copies.
 * The input dimensions must match a mode supported by the selected connector.
 */
class DrmDisplayOp final : public holoscan::Operator<> {
 public:
  DrmDisplayOp(int width,
               int height,
               std::string device = "/dev/dri/card0",
               std::uint32_t connector_id = 0U,
               std::string mode = {});
  ~DrmDisplayOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;

 private:
  struct Impl;

  int width_{};
  int height_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops
