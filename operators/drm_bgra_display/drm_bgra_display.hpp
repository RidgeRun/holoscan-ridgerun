// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
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
 * @brief Display CUDA-device BGRA tensors through an exclusive DRM/KMS output.
 *
 * This variant accepts the BGRA tensor produced by CudaCompositorOp. It uses
 * direct CUDA DMABUF import where supported, a CUDA-registered DRM mapping on
 * Orin where available, and a direct mmap copy as a compatibility fallback.
 * It does not use X11 or Wayland.
 */
class DrmBgraDisplayOp final : public holoscan::Operator<> {
 public:
  DrmBgraDisplayOp(int width,
                   int height,
                   std::string device = "/dev/dri/card0",
                   std::uint32_t connector_id = 0U,
                   std::string mode = {});
  ~DrmBgraDisplayOp() override;

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
  std::size_t frame_bytes_{};
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops
