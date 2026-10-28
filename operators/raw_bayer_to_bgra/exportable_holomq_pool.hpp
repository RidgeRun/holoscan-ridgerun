// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/tensor_contract.hpp>

namespace holoscan::ops {

class ExportableHoloMQPool final {
 public:
  ExportableHoloMQPool(std::size_t buffer_bytes,
                       std::uint32_t buffer_count,
                       std::int32_t cuda_device);
  ~ExportableHoloMQPool();

  ExportableHoloMQPool(const ExportableHoloMQPool&) = delete;
  ExportableHoloMQPool& operator=(const ExportableHoloMQPool&) = delete;

  [[nodiscard]] holoscan::expected<holoscan::Tensor, holoscan::Error> acquire_tensor(
      holoscan::TensorShapeView shape, DLDataType dtype);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace holoscan::ops
