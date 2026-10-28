// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <memory>
#include <string>

#include "interprocess_src/interprocess_reader.hpp"

namespace holoscan::ops {

/// Shared end-of-stream state used by the source operator and receiver application.
class InterprocessEndOfStream final {
 public:
  InterprocessEndOfStream() : signaled_(std::make_shared<std::atomic_bool>(false)) {}

  void signal() noexcept { signaled_->store(true, std::memory_order_release); }
  [[nodiscard]] bool signaled() const noexcept {
    return signaled_->load(std::memory_order_acquire);
  }

 private:
  std::shared_ptr<std::atomic_bool> signaled_;
};

/// Shared state raised after InterprocessSrc has created every configured subscriber.
class InterprocessReceiverReady final {
 public:
  InterprocessReceiverReady() : signaled_(std::make_shared<std::atomic_bool>(false)) {}

  void signal() noexcept { signaled_->store(true, std::memory_order_release); }
  [[nodiscard]] bool signaled() const noexcept {
    return signaled_->load(std::memory_order_acquire);
  }

 private:
  std::shared_ptr<std::atomic_bool> signaled_;
};

/// Read one external Tensor endpoint as a graph source.
class InterprocessSrcOp final : public holoscan::Operator<> {
 public:
  InterprocessSrcOp(holoscan::ChannelEndpointCatalog catalog, std::string endpoint_name,
                    std::chrono::nanoseconds poll_period = std::chrono::milliseconds{5},
                    InterprocessEndOfStream end_of_stream = InterprocessEndOfStream{},
                    InterprocessReceiverReady receiver_ready = InterprocessReceiverReady{},
                    bool retain_imported_tensor = false);

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Output<holoscan::Tensor> output;

 private:
  holoscan::ChannelEndpointCatalog catalog_;
  std::string endpoint_name_;
  std::chrono::nanoseconds poll_period_{};
  std::uint32_t output_max_rank_{};
  std::uint64_t output_max_byte_span_{};
  InterprocessEndOfStream end_of_stream_;
  std::unique_ptr<holoscan::ChannelRuntime> runtime_;
  InterprocessReceiverReady receiver_ready_;
  bool retain_imported_tensor_{};
  std::unique_ptr<InterprocessReader> reader_;
};

}  // namespace holoscan::ops
