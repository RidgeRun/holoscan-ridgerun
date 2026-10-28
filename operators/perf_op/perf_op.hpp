// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>

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
 * @brief Zero-copy Tensor pass-through that periodically reports throughput.
 *
 * The data path never reads or copies the Tensor payload. It accounts the
 * Tensor's logical byte span and forwards the complete owning Sample<Tensor>,
 * preserving its storage and semantic metadata.
 */
class PerfOp final : public holoscan::Operator<> {
 public:
  explicit PerfOp(std::uint32_t bitrate_interval_ms = 1000U,
                  std::uint32_t bitrate_window_size = 0U,
                  bool print_cpu_load = false);
  PerfOp(holoscan::TensorRepresentation representation,
         std::uint32_t bitrate_interval_ms = 1000U,
         std::uint32_t bitrate_window_size = 0U,
         bool print_cpu_load = false);
  ~PerfOp() override;

  PerfOp(const PerfOp&) = delete;
  PerfOp& operator=(const PerfOp&) = delete;
  PerfOp(PerfOp&&) = delete;
  PerfOp& operator=(PerfOp&&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  struct CpuCounters {
    std::uint64_t idle{};
    std::uint64_t total{};
    bool valid{};
  };

  [[nodiscard]] static CpuCounters read_cpu_counters() noexcept;
  void report_loop();

  std::uint32_t bitrate_interval_ms_{};
  std::uint32_t bitrate_window_size_{};
  bool print_cpu_load_{};
  std::optional<holoscan::TensorRepresentation> representation_;

  std::atomic<std::uint64_t> interval_frames_{0U};
  std::atomic<std::uint64_t> interval_bytes_{0U};
  std::mutex reporter_mutex_;
  std::condition_variable reporter_cv_;
  bool stop_reporter_{true};
  std::thread reporter_;
};

}  // namespace holoscan::ops
