// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file interprocess_sender.cpp
 * @brief Publish a live V4L2 stream through an exported Holoscan SDK (HSDK) 5 endpoint.
 */

#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "frame_limit/frame_limit.hpp"
#include "raw_bayer_to_bgra/raw_bayer_to_bgra.hpp"
#include "v4l2_userptr_source/v4l2_userptr_source.hpp"

namespace {

using holoscan::ops::FrameLimit;
using holoscan::ops::FrameStartGate;
using holoscan::ops::RawBayerToBgraOp;
using holoscan::ops::V4L2UserPtrSourceOp;

constexpr std::string_view kEndpoint{"video"};
constexpr std::string_view kReadyEndpoint{"receiver_ready"};
constexpr std::string_view kDoneEndpoint{"sender_done"};
constexpr std::string_view kUriPrefix{"holomq://rr/interprocess/"};

volatile std::sig_atomic_t g_stop_requested = 0;

using ReceiverStartSignal = FrameStartGate;

class ReceiverReadyOp final : public holoscan::Operator<> {
 public:
  explicit ReceiverReadyOp(ReceiverStartSignal signal) : signal_(std::move(signal)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto value = input.receive_data();
    if (!value) {
      return holoscan::make_unexpected(std::move(value).error());
    }
    if (*value != 0U) {
      signal_.open();
    }
    return {};
  }

  holoscan::Input<std::uint32_t> input;

 private:
  ReceiverStartSignal signal_;
};

class SenderDoneSignal final {
 public:
  SenderDoneSignal() : sent_(std::make_shared<std::atomic_bool>(false)) {}

  void signal() noexcept { sent_->store(true, std::memory_order_release); }
  [[nodiscard]] bool signaled() const noexcept {
    return sent_->load(std::memory_order_acquire);
  }

 private:
  std::shared_ptr<std::atomic_bool> sent_;
};

class SenderDoneOp final : public holoscan::Operator<> {
 public:
  SenderDoneOp(FrameLimit frame_limit, SenderDoneSignal done)
      : frame_limit_(std::move(frame_limit)), done_(std::move(done)) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output").max_emits_per_compute(1U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    if (!frame_limit_.reached() || done_.signaled()) {
      return {};
    }
    auto emitted = output.emit(1U);
    if (emitted) {
      done_.signal();
    }
    return emitted;
  }

  holoscan::Output<std::uint32_t> output;

 private:
  FrameLimit frame_limit_;
  SenderDoneSignal done_;
};

struct Options {
  std::filesystem::path catalog;
  std::string device{"/dev/video0"};
  std::int32_t width{1920};
  std::int32_t height{1080};
  std::int32_t fps{60};
  std::int32_t cuda_device{0};
  std::int64_t sensor_gain{8'000'000};
  std::int64_t exposure_us{15'649};
  std::int32_t optical_black{50};
  std::int32_t raw_shift{6};
  float display_gain{12.0F};
  float gamma{2.2F};
  float wb_red{0.833F};
  float wb_green{1.0F};
  float wb_blue{0.833F};
  std::int32_t bayer_grid{1};
  std::uint32_t duration_seconds{};
  std::uint64_t frame_limit{};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " --catalog PATH [options]\n\n"
         << "Publish a live V4L2 USERPTR stream for the interprocess demo.\n\n"
         << "Options:\n"
         << "  --catalog PATH       Catalog file shared with interprocess_receiver\n"
         << "  --device PATH        V4L2 capture device (default: /dev/video0)\n"
         << "  --width PIXELS       Exact RG10 capture width (default: 1920)\n"
         << "  --height PIXELS      Exact RG10 capture height (default: 1080)\n"
         << "  --fps RATE           Exact capture rate (default: 60)\n"
         << "  --cuda-device INDEX  CUDA device ordinal (default: 0)\n"
         << "  --sensor-gain VALUE  Tegra sensor gain in micro-units (default: 8000000)\n"
         << "  --exposure-us VALUE  Tegra exposure in microseconds (default: 15649)\n"
         << "  --optical-black N    RAW10 black-level offset (default: 50)\n"
         << "  --raw-shift N        Right shift for RAW10 samples (default: 6)\n"
         << "  --display-gain X     Digital gain after black level (default: 12.0)\n"
         << "  --gamma X            Display gamma encoding (default: 2.2)\n"
         << "  --wb-red X           Fixed red WB gain; all three 0 selects gray-world\n"
         << "  --wb-green X         Fixed green WB gain\n"
         << "  --wb-blue X          Fixed blue WB gain\n"
         << "  --bayer-grid N       0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG (default: 1)\n"
         << "  --duration SECONDS   Stop after a fixed duration (default: run until stopped)\n"
         << "  --frame-limit COUNT  Stop after synchronized published frames (default: unlimited)\n"
         << "  --validate            Compile and save the catalog, but do not publish frames\n"
         << "  -h, --help           Show this help text\n";
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] float parse_float(std::string_view text, std::string_view option) {
  float value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires a number");
  }
  return value;
}

[[nodiscard]] std::string_view option_value(int& index, int argc, char** argv,
                                            std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[++index];
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "-h" || argument == "--help") {
      options.show_help = true;
    } else if (argument == "--catalog") {
      options.catalog = option_value(index, argc, argv, argument);
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--cuda-device") {
      options.cuda_device =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--sensor-gain") {
      options.sensor_gain =
          parse_integer<std::int64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--exposure-us") {
      options.exposure_us =
          parse_integer<std::int64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--optical-black") {
      options.optical_black =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--raw-shift") {
      options.raw_shift =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--display-gain") {
      options.display_gain = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--gamma") {
      options.gamma = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--wb-red") {
      options.wb_red = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--wb-green") {
      options.wb_green = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--wb-blue") {
      options.wb_blue = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--bayer-grid") {
      options.bayer_grid =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--duration") {
      options.duration_seconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--frame-limit") {
      options.frame_limit =
          parse_integer<std::uint64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--validate") {
      options.validate_only = true;
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (options.show_help) {
    return options;
  }
  if (options.catalog.empty()) {
    throw std::invalid_argument("--catalog is required");
  }
  if (options.device.empty()) {
    throw std::invalid_argument("--device must not be empty");
  }
  if (options.width <= 0 || options.height <= 0 || options.fps <= 0 ||
      options.cuda_device < 0 || options.sensor_gain <= 0 || options.exposure_us <= 0) {
    throw std::invalid_argument(
        "capture dimensions, fps, sensor gain, and exposure must be positive; "
        "--cuda-device must be nonnegative");
  }
  if ((options.width & 1) != 0 || (options.height & 1) != 0) {
    throw std::invalid_argument("--width and --height must be even for Bayer capture");
  }
  if (options.optical_black < 0 || options.optical_black >= 1023) {
    throw std::invalid_argument("--optical-black must be in [0,1022]");
  }
  if (options.raw_shift < 0 || options.raw_shift > 6) {
    throw std::invalid_argument("--raw-shift must be in [0,6]");
  }
  if (!std::isfinite(options.display_gain) || options.display_gain <= 0.0F ||
      options.display_gain > 64.0F) {
    throw std::invalid_argument("--display-gain must be finite and in (0,64]");
  }
  if (!std::isfinite(options.gamma) || options.gamma <= 0.0F || options.gamma > 5.0F) {
    throw std::invalid_argument("--gamma must be finite and in (0,5]");
  }
  const bool dynamic_wb = options.wb_red == 0.0F && options.wb_green == 0.0F &&
                          options.wb_blue == 0.0F;
  const bool fixed_wb = std::isfinite(options.wb_red) && std::isfinite(options.wb_green) &&
                        std::isfinite(options.wb_blue) && options.wb_red > 0.0F &&
                        options.wb_green > 0.0F && options.wb_blue > 0.0F &&
                        options.wb_red <= 16.0F && options.wb_green <= 16.0F &&
                        options.wb_blue <= 16.0F;
  if (!dynamic_wb && !fixed_wb) {
    throw std::invalid_argument(
        "WB gains must either all be 0 (gray-world) or all be finite values in (0,16]");
  }
  if (options.bayer_grid < 0 || options.bayer_grid > 3) {
    throw std::invalid_argument("--bayer-grid must be in [0,3]");
  }
  return options;
}

[[nodiscard]] holoscan::TopicOptions topic_options(std::string_view endpoint_name,
                                                   std::int32_t width, std::int32_t height,
                                                   holoscan::ClockKey capture_clock) {
  holoscan::TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = holoscan::RequireCaptureTime{.clock = capture_clock};
  options.attach.peer = holoscan::PeerRequirement::kOptional;
  options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 2U,
      .max_borrowed_samples = 2U,
  };
  options.tensor = holoscan::TensorBoundaryContract{
      .representation =
          {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 3U,
              .strides = holoscan::TensorStridePolicy::kContiguousOnly,
          },
      .bounds =
          holoscan::TensorBounds{
              .max_rank = 3U,
              .max_byte_span =
                  static_cast<std::uint64_t>(width) * static_cast<std::uint64_t>(height) * 4U,
          },
      .transfer = holoscan::TensorTransferPolicy{
          .foreign_emit_policy = holoscan::InterfaceTensorEmitPolicy::kRequireExportable,
      },
  };
  return options;
}

void bind_tensor_output(holoscan::CompileOptions& compile_options, std::string_view operator_path,
                        std::string_view output_port,
                        std::int32_t cuda_device) {
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = operator_path,
      .output_port = output_port,
      .device = holoscan::DeviceId{cuda_device},
  });
}

[[nodiscard]] holoscan::ExecutionPlan build_plan(const Options& options,
                                                 FrameLimit frame_limit,
                                                 ReceiverStartSignal start_signal,
                                                 SenderDoneSignal done_signal) {
  holoscan::Graph graph{"interprocess-sender"};
  const auto runtime_clock = graph.add_clock<holoscan::RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  const auto camera = graph.op<V4L2UserPtrSourceOp>("camera",
                                                    options.device,
                                                    options.width,
                                                    options.height,
                                                    options.fps,
                                                    options.cuda_device,
                                                    options.sensor_gain,
                                                    options.exposure_us,
                                                    frame_limit,
                                                    start_signal);

  holoscan::TopicOptions ready_options;
  ready_options.endpoint_name = kReadyEndpoint;
  ready_options.bounds.publisher =
      holoscan::PublisherEndpointBounds{.max_loaned_samples = 1U};
  const auto receiver_ready = graph.import_topic<std::uint32_t>(
      std::string{kUriPrefix} + std::string{kReadyEndpoint}, std::move(ready_options));
  const auto ready_op = graph.op<ReceiverReadyOp>("receiver_ready", start_signal);
  const auto done_op = graph.op<SenderDoneOp>("sender_done", frame_limit, done_signal);
  std::optional<std::array<float, 3U>> fixed_white_balance;
  if (options.wb_red != 0.0F || options.wb_green != 0.0F || options.wb_blue != 0.0F) {
    fixed_white_balance = std::array<float, 3U>{
        options.wb_red, options.wb_green, options.wb_blue};
  }
  const auto convert = graph.op<RawBayerToBgraOp>("convert",
                                                  options.width,
                                                  options.height,
                                                  options.optical_black,
                                                  options.bayer_grid,
                                                  options.cuda_device,
                                                  options.raw_shift,
                                                  options.display_gain,
                                                  options.gamma,
                                                  fixed_white_balance,
                                                  true);

  const auto exported = graph.export_topic<holoscan::Tensor>(
      std::string{kUriPrefix} + std::string{kEndpoint},
      topic_options(kEndpoint, options.width, options.height, runtime_clock));

  holoscan::TopicOptions done_options;
  done_options.endpoint_name = kDoneEndpoint;
  done_options.bounds.subscriber = holoscan::SubscriberEndpointBounds{
      .queue_depth = 1U,
      .max_borrowed_samples = 1U,
  };
  const auto sender_done = graph.export_topic<std::uint32_t>(
      std::string{kUriPrefix} + std::string{kDoneEndpoint}, std::move(done_options));

  const holoscan::ConnectionOptions latest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  const holoscan::ConnectionOptions export_queue{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 2U,
  };
  graph.add_flow(receiver_ready.output(), ready_op->input,
                 holoscan::ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(camera->frame, convert->input, latest_one);
  graph.add_flow(convert->output, exported.input(), export_queue);
  graph.add_flow(done_op->output, sender_done.input(),
                 holoscan::ConnectionOptions{.queue_depth = 1U});

  holoscan::CompileOptions compile_options;
  bind_tensor_output(compile_options, "convert", "output", options.cuda_device);
  return holoscan::compile(graph, std::move(compile_options));
}

[[nodiscard]] int run(const Options& options) {
  FrameLimit frame_limit{options.frame_limit};
  ReceiverStartSignal start_signal{false};
  SenderDoneSignal done_signal;
  const holoscan::ExecutionPlan plan =
      build_plan(options, frame_limit, start_signal, done_signal);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.catalog.has_parent_path()) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(options.catalog.parent_path(), filesystem_error);
    if (filesystem_error) {
      std::cerr << "interprocess_sender: catalog directory creation failed: "
                << filesystem_error.message() << '\n';
      return 1;
    }
  }
  auto saved = plan.endpoint_catalog().save_json(options.catalog);
  if (!saved) {
    std::cerr << "interprocess_sender: catalog save failed: "
              << holoscan::runtime_error_message(saved.error()) << '\n';
    return 1;
  }
  std::cout << "interprocess_sender catalog written to " << options.catalog << '\n';
  if (options.validate_only) {
    std::cout << "interprocess_sender graph validation complete\n";
    return 0;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  holoscan::RunSession session = holoscan::run_async(plan);
  auto deadline = std::chrono::steady_clock::time_point::max();
  bool receiver_announced = false;
  std::cout << "interprocess_sender: waiting for receiver readiness\n";
  while (session.is_running() && g_stop_requested == 0 &&
         (!frame_limit.reached() || !done_signal.signaled())) {
    if (!receiver_announced && start_signal.is_open()) {
      receiver_announced = true;
      if (options.duration_seconds != 0U) {
        deadline = std::chrono::steady_clock::now() +
                   std::chrono::seconds{options.duration_seconds};
      }
      std::cout << "interprocess_sender: receiver ready; publishing frames\n";
      std::cout.flush();
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      g_stop_requested = 1;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }
  if (session.is_running() && frame_limit.reached() && done_signal.signaled()) {
    // Allow the receiver to release its final imported buffers before the endpoint pools are
    // destroyed. No additional frames are captured after FrameLimit is reached.
    std::this_thread::sleep_for(std::chrono::seconds{2});
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::cerr << "interprocess_sender diagnostic: " << diagnostic.token << '\n';
  }
  if (session.termination() == holoscan::RunTermination::kFailed ||
      !session.diagnostics().empty()) {
    return 2;
  }
  std::cout << "interprocess_sender stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "interprocess_sender");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "interprocess_sender: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "interprocess_sender");
    return 64;
  }
}
