// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <optional>
#include <string>

namespace holoscan::ops {

/// Own one cross-process Tensor subscriber for its complete lifetime.
class InterprocessReader final {
 public:
  InterprocessReader(holoscan::ChannelRuntime& runtime, holoscan::ChannelEndpointCatalog catalog,
                     std::string endpoint_name);

  InterprocessReader(const InterprocessReader&) = delete;
  InterprocessReader& operator=(const InterprocessReader&) = delete;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> start();
  void stop() noexcept;
  [[nodiscard]] holoscan::expected<std::optional<holoscan::Sample<holoscan::Tensor>>,
                                   holoscan::Error>
  receive();

 private:
  holoscan::ChannelRuntime* runtime_{};
  holoscan::ChannelEndpointCatalog catalog_;
  std::string endpoint_name_;
  std::optional<holoscan::ChannelSubscriber<holoscan::Tensor>> subscriber_;
};

}  // namespace holoscan::ops
