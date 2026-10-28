// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "perf_op/perf_op.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <syncstream>
#include <utility>

namespace holoscan::ops {

PerfOp::PerfOp(std::uint32_t bitrate_interval_ms,
               std::uint32_t bitrate_window_size,
               bool print_cpu_load)
    : bitrate_interval_ms_(bitrate_interval_ms),
      bitrate_window_size_(bitrate_window_size),
      print_cpu_load_(print_cpu_load) {
  if (bitrate_interval_ms_ == 0U) {
    throw std::invalid_argument("PerfOp bitrate interval must be greater than zero");
  }
}

PerfOp::PerfOp(holoscan::TensorRepresentation representation,
               std::uint32_t bitrate_interval_ms,
               std::uint32_t bitrate_window_size,
               bool print_cpu_load)
    : PerfOp(bitrate_interval_ms, bitrate_window_size, print_cpu_load) {
  representation_ = std::move(representation);
}

PerfOp::~PerfOp() { stop(); }

void PerfOp::setup(holoscan::OperatorSpec& spec) {
  auto& input_builder = spec.input(input, "input").queue_depth(1U);
  auto& output_builder = spec.output(output, "output").max_emits_per_compute(1U);
  if (representation_) {
    input_builder.expects_tensor(
        holoscan::TensorInputSpec{.representation = *representation_});
    output_builder.produces_tensor(
        holoscan::TensorOutputSpec{
            .representation = *representation_,
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        },
        holoscan::input_max_byte_span(input));
  }
}

holoscan::Contract PerfOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

void PerfOp::start() {
  stop();
  interval_frames_.store(0U, std::memory_order_relaxed);
  interval_bytes_.store(0U, std::memory_order_relaxed);
  {
    const std::lock_guard lock{reporter_mutex_};
    stop_reporter_ = false;
  }
  reporter_ = std::thread(&PerfOp::report_loop, this);
}

void PerfOp::stop() {
  {
    const std::lock_guard lock{reporter_mutex_};
    stop_reporter_ = true;
  }
  reporter_cv_.notify_all();
  if (reporter_.joinable()) {
    reporter_.join();
  }
}

holoscan::expected<void, holoscan::Error> PerfOp::compute(
    holoscan::ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  // nbytes() consults Tensor metadata only. No payload access, CUDA call,
  // allocation, synchronization, or copy occurs on this path.
  interval_bytes_.fetch_add(sample->data.nbytes(), std::memory_order_relaxed);
  interval_frames_.fetch_add(1U, std::memory_order_relaxed);
  return output.emit(std::move(*sample));
}

PerfOp::CpuCounters PerfOp::read_cpu_counters() noexcept {
  std::ifstream stat{"/proc/stat"};
  std::string cpu;
  std::uint64_t user{};
  std::uint64_t nice{};
  std::uint64_t system{};
  std::uint64_t idle{};
  std::uint64_t iowait{};
  std::uint64_t irq{};
  std::uint64_t softirq{};
  std::uint64_t steal{};
  if (!(stat >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) ||
      cpu != "cpu") {
    return {};
  }
  return CpuCounters{
      .idle = idle + iowait,
      .total = user + nice + system + idle + iowait + irq + softirq + steal,
      .valid = true,
  };
}

void PerfOp::report_loop() {
  using Clock = std::chrono::steady_clock;
  const auto interval = std::chrono::milliseconds{bitrate_interval_ms_};
  auto previous_time = Clock::now();
  CpuCounters previous_cpu = print_cpu_load_ ? read_cpu_counters() : CpuCounters{};
  std::uint64_t report_count{};
  double mean_fps{};
  double mean_bitrate{};
  std::deque<double> bitrate_window;

  std::unique_lock lock{reporter_mutex_};
  while (!stop_reporter_) {
    if (reporter_cv_.wait_for(lock, interval, [this] { return stop_reporter_; })) {
      break;
    }
    lock.unlock();

    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - previous_time).count();
    previous_time = now;
    const std::uint64_t frames = interval_frames_.exchange(0U, std::memory_order_relaxed);
    const std::uint64_t bytes = interval_bytes_.exchange(0U, std::memory_order_relaxed);
    const double fps = elapsed > 0.0 ? static_cast<double>(frames) / elapsed : 0.0;
    const double bitrate = elapsed > 0.0
                               ? static_cast<double>(bytes) * 8.0 / elapsed
                               : 0.0;

    ++report_count;
    mean_fps += (fps - mean_fps) / static_cast<double>(report_count);
    if (bitrate_window_size_ == 0U) {
      mean_bitrate += (bitrate - mean_bitrate) / static_cast<double>(report_count);
    } else {
      bitrate_window.push_back(bitrate);
      if (bitrate_window.size() > bitrate_window_size_) {
        bitrate_window.pop_front();
      }
      mean_bitrate = std::accumulate(bitrate_window.begin(), bitrate_window.end(), 0.0) /
                     static_cast<double>(bitrate_window.size());
    }

    double cpu_load{};
    bool cpu_valid{};
    if (print_cpu_load_) {
      const CpuCounters current_cpu = read_cpu_counters();
      if (previous_cpu.valid && current_cpu.valid && current_cpu.total > previous_cpu.total) {
        const std::uint64_t total_delta = current_cpu.total - previous_cpu.total;
        const std::uint64_t idle_delta = current_cpu.idle - previous_cpu.idle;
        cpu_load = 100.0 * static_cast<double>(total_delta - std::min(idle_delta, total_delta)) /
                   static_cast<double>(total_delta);
        cpu_valid = true;
      }
      previous_cpu = current_cpu;
    }

    std::osyncstream message{std::cout};
    message << std::fixed << std::setprecision(3)
            << "perf: fps: " << fps
            << "; mean_fps: " << mean_fps
            << "; bps: " << bitrate
            << "; mean_bps: " << mean_bitrate;
    if (print_cpu_load_) {
      if (cpu_valid) {
        message << "; cpu: " << std::setprecision(1) << cpu_load << '%';
      } else {
        message << "; cpu: unavailable";
      }
    }
    message << ';' << '\n';

    lock.lock();
  }
}

}  // namespace holoscan::ops
