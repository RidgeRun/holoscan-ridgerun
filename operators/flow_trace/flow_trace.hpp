// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <unistd.h>

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
 * @brief Parser-compatible software trace for the Holoscan SDK (HSDK) 5 Graph API.
 *
 * The HoloHub Flow benchmark expects the legacy DataFlowTracker text format,
 * while the HSDK 5 Graph/ExecutionPlan API does not expose that tracker.
 * This adapter records the three operators in this sample using the frame_id
 * already carried by SampleMetadata and writes one complete path at the sink.
 *
 * The timestamps are CLOCK_MONOTONIC-equivalent host timestamps in microseconds.
 * They represent software receipt/publication and display submission; they are
 * not a physical glass-to-glass measurement.
 */
class FlowTrace final {
 public:
  static FlowTrace& instance() {
    static FlowTrace trace;
    return trace;
  }

  [[nodiscard]] static std::int64_t now_microseconds() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::microseconds>(now).count();
  }

  void record(std::uint64_t frame_id,
              std::string_view operator_name,
              std::int64_t receive_us,
              std::int64_t publish_us) {
    if (frame_id == 0U) {
      return;
    }

    std::lock_guard lock(mutex_);
    const std::int64_t ordered_publish_us = publish_us < receive_us ? receive_us : publish_us;
    if (event_mode_) {
      event_file_ << frame_id << ',' << operator_name << ',' << receive_us << ','
                  << ordered_publish_us << std::endl;
      return;
    }
    if (!enabled_) {
      return;
    }

    TraceEntry& entry = entries_[frame_id];

    if (operator_name == "camera") {
      entry.camera_receive_us = receive_us;
      entry.camera_publish_us = ordered_publish_us;
      entry.has_camera = true;
    } else if (operator_name == "convert") {
      entry.convert_receive_us = receive_us;
      entry.convert_publish_us = ordered_publish_us;
      entry.has_convert = true;
    } else if (operator_name == "preprocess") {
      entry.preprocess_receive_us = receive_us;
      entry.preprocess_publish_us = ordered_publish_us;
      entry.has_preprocess = true;
    } else if (operator_name == "display") {
      entry.display_receive_us = receive_us;
      entry.display_publish_us = ordered_publish_us;
      entry.has_display = true;
    } else if (operator_name == "inference") {
      entry.inference_receive_us = receive_us;
      entry.inference_publish_us = ordered_publish_us;
      entry.has_inference = true;
    } else if (operator_name == "sink") {
      entry.sink_receive_us = receive_us;
      entry.sink_publish_us = ordered_publish_us;
      entry.has_sink = true;
    }

    if (entry.has_camera && (entry.has_display || entry.has_sink)) {
      log_path(frame_id, entry);
      entries_.erase(frame_id);
    }

    // A dropped frame may never reach the display. Bound memory in that case.
    if (entries_.size() > 4096U) {
      entries_.erase(entries_.begin());
    }
  }

 private:
  struct TraceEntry {
    std::int64_t camera_receive_us{};
    std::int64_t camera_publish_us{};
    std::int64_t convert_receive_us{};
    std::int64_t convert_publish_us{};
    std::int64_t preprocess_receive_us{};
    std::int64_t preprocess_publish_us{};
    std::int64_t inference_receive_us{};
    std::int64_t inference_publish_us{};
    std::int64_t sink_receive_us{};
    std::int64_t sink_publish_us{};
    std::int64_t display_receive_us{};
    std::int64_t display_publish_us{};
    bool has_camera{};
    bool has_convert{};
    bool has_preprocess{};
    bool has_inference{};
    bool has_sink{};
    bool has_display{};
  };

  FlowTrace() {
    const char* const path = std::getenv("HOLOSCAN_FLOW_TRACKING_LOG_FILE");
    if (path == nullptr || *path == '\0') {
      return;
    }
    const char* const cross_process = std::getenv("HOLOSCAN_FLOW_CROSS_PROCESS_EVENTS");
    if (cross_process != nullptr && std::string_view{cross_process} == "1") {
      const std::string event_path =
          std::string{path} + "." + std::to_string(static_cast<long long>(::getpid())) +
          ".events";
      event_file_.open(event_path, std::ios::out | std::ios::trunc);
      event_mode_ = event_file_.is_open();
      if (!event_mode_) {
        std::cerr << "[flow_trace] unable to open event log file: "
                  << event_path << '\n';
      }
      return;
    }

    log_file_.open(path, std::ios::out | std::ios::trunc);
    enabled_ = log_file_.is_open();
    if (!enabled_) {
      std::cerr << "[flow_trace] unable to open log file: " << path << '\n';
    }
  }

  ~FlowTrace() {
    std::lock_guard lock(mutex_);
    if (log_file_.is_open()) {
      log_file_.flush();
      log_file_.close();
    }
    if (event_file_.is_open()) {
      event_file_.flush();
      event_file_.close();
    }
  }

  void log_path(std::uint64_t frame_id, const TraceEntry& entry) {
    log_file_ << "(camera," << entry.camera_receive_us << ',' << entry.camera_publish_us;
    if (entry.has_convert) {
      log_file_ << ") -> (convert," << entry.convert_receive_us << ',' << entry.convert_publish_us;
    }
    if (entry.has_preprocess) {
      log_file_ << ") -> (preprocess," << entry.preprocess_receive_us << "," << entry.preprocess_publish_us;
    }
    if (entry.has_inference) {
      log_file_ << ") -> (inference," << entry.inference_receive_us << "," << entry.inference_publish_us;
    }
    if (entry.has_display) {
      log_file_ << ") -> (display," << entry.display_receive_us << "," << entry.display_publish_us << ")" << std::endl;
    } else if (entry.has_sink) {
      log_file_ << ") -> (sink," << entry.sink_receive_us << "," << entry.sink_publish_us << ")" << std::endl;
    }
    // Keep the file visible to the outer runner even if the process is stopped
    // cooperatively before normal static destruction.
    if ((frame_id % 32U) == 0U) {
      log_file_.flush();
    }
  }

  std::ofstream log_file_;
  std::ofstream event_file_;
  bool enabled_{};
  bool event_mode_{};
  std::mutex mutex_;
  std::unordered_map<std::uint64_t, TraceEntry> entries_;
};

/// Zero-copy Tensor pass-through that records one named flow stage.
class FlowTraceOp final : public holoscan::Operator<> {
 public:
  explicit FlowTraceOp(std::string stage_name) : stage_name_(std::move(stage_name)) {
    if (stage_name_.empty()) {
      throw std::invalid_argument("FlowTraceOp stage name must not be empty");
    }
  }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    const std::int64_t receive_us = FlowTrace::now_microseconds();
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    const std::uint64_t frame_id = sample->metadata.frame_id;
    auto emitted = output.emit(std::move(*sample));
    if (!emitted) {
      return emitted;
    }
    FlowTrace::instance().record(
        frame_id, stage_name_, receive_us, FlowTrace::now_microseconds());
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  std::string stage_name_;
};

}  // namespace holoscan::ops
