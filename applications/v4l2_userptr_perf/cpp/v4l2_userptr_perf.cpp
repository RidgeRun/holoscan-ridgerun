// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * V4L2 USERPTR pipeline with a PerfOp Tensor pass-through:
 *
 * V4L2 RG10 -> CUDA Bayer-to-BGRA -> PerfOp -> X11
 */

#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "frame_limit/frame_limit.hpp"
#include "perf_op/perf_op.hpp"
#include "raw_bayer_to_bgra/raw_bayer_to_bgra.hpp"
#include "v4l2_userptr_source/v4l2_userptr_source.hpp"
#include "operator_utils/tensor_utils.hpp"
#include "x11_display/x11_display.hpp"

namespace {

using holoscan::ops::FrameLimit;
using holoscan::ops::RawBayerToBgraOp;
using holoscan::ops::V4L2UserPtrSourceOp;
using holoscan::ops::X11DisplayOp;
using holoscan::ops::PerfOp;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::string device{"/dev/video0"};
  std::int32_t width{1920};
  std::int32_t height{1080};
  std::int32_t fps{60};
  std::int64_t sensor_gain{8'000'000};
  std::int64_t exposure_us{15'649};
  std::uint32_t duration_seconds{};
  std::uint64_t frame_limit{};
  std::int32_t cuda_device{0};
  std::int32_t optical_black{50};
  std::int32_t raw_shift{6};
  float display_gain{12.0F};
  float gamma{2.2F};
  std::int32_t bayer_grid{1};
  std::uint32_t bitrate_interval_ms{1000U};
  std::uint32_t bitrate_window_size{};
  bool print_cpu_load{};
  bool dump_plan{};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n\n"
      << "V4L2 USERPTR capture, CUDA Bayer processing, performance pass-through, and X11.\n\n"
      << "Capture and display options:\n"
      << "  --device PATH             V4L2 device (default: /dev/video0)\n"
      << "  --width PIXELS            RG10 width (default: 1920)\n"
      << "  --height PIXELS           RG10 height (default: 1080)\n"
      << "  --fps RATE                Capture rate (default: 60)\n"
      << "  --sensor-gain VALUE       Sensor gain in micro-units (default: 8000000)\n"
      << "  --exposure-us VALUE       Exposure in microseconds (default: 15649)\n"
      << "  --duration SECONDS        Stop after a duration (default: unlimited)\n"
      << "  --frame-limit COUNT       Stop after captured frames (default: unlimited)\n"
      << "  --cuda-device INDEX       CUDA device ordinal (default: 0)\n"
      << "  --optical-black N         RAW10 black level (default: 50)\n"
      << "  --raw-shift N             RAW10 right shift (default: 6)\n"
      << "  --display-gain X          Digital display gain (default: 12.0)\n"
      << "  --gamma X                 Display gamma (default: 2.2)\n"
      << "  --bayer-grid N            0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG (default: 1)\n\n"
      << "Performance options:\n"
      << "  --bitrate-interval MS     Report interval (default: 1000)\n"
      << "  --bitrate-window-size N   Bitrate moving-average samples; 0=all (default: 0)\n"
      << "  --print-cpu-load          Print aggregate system CPU load\n"
      << "  --print-arm-load          Deprecated alias for --print-cpu-load\n"
      << "  --dump-plan              Print the compiled execution plan as JSON\n"
      << "  --validate                Compile without opening camera/display\n"
      << "  -h, --help                Show this help text\n\n"
      << "Bitrate describes BGRA Tensor bytes at PerfOp, not encoded video bitrate.\n";
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const auto [next, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || next != text.data() + text.size()) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] float parse_float(std::string_view text, std::string_view option) {
  float value{};
  const auto [next, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || next != text.data() + text.size()) {
    throw std::invalid_argument(std::string(option) + " requires a number");
  }
  return value;
}

[[nodiscard]] std::string_view option_value(
    int& index, int argc, char** argv, std::string_view option) {
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
    } else if (argument == "--validate") {
      options.validate_only = true;
    } else if (argument == "--dump-plan") {
      options.dump_plan = true;
    } else if (argument == "--print-cpu-load" || argument == "--print-arm-load") {
      if (argument == "--print-arm-load") {
        std::cerr << "warning: --print-arm-load is deprecated; use --print-cpu-load\n";
      }
      options.print_cpu_load = true;
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--sensor-gain") {
      options.sensor_gain =
          parse_integer<std::int64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--exposure-us") {
      options.exposure_us =
          parse_integer<std::int64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--duration") {
      options.duration_seconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--frame-limit") {
      options.frame_limit =
          parse_integer<std::uint64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--cuda-device") {
      options.cuda_device =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
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
    } else if (argument == "--bayer-grid") {
      options.bayer_grid =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--bitrate-interval") {
      options.bitrate_interval_ms =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--bitrate-window-size") {
      options.bitrate_window_size =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.device.empty() || options.width <= 0 || options.height <= 0 || options.fps <= 0 ||
      options.sensor_gain <= 0 || options.exposure_us <= 0 || options.cuda_device < 0) {
    throw std::invalid_argument("capture dimensions, rate, gain, and exposure must be positive");
  }
  if ((options.width & 1) != 0 || (options.height & 1) != 0) {
    throw std::invalid_argument("--width and --height must be even for Bayer capture");
  }
  if (options.optical_black < 0 || options.optical_black >= 1023 ||
      options.raw_shift < 0 || options.raw_shift > 6 ||
      options.bayer_grid < 0 || options.bayer_grid > 3) {
    throw std::invalid_argument("invalid RAW Bayer processing option");
  }
  if (!std::isfinite(options.display_gain) || options.display_gain <= 0.0F ||
      !std::isfinite(options.gamma) || options.gamma <= 0.0F) {
    throw std::invalid_argument("--display-gain and --gamma must be positive finite values");
  }
  if (options.bitrate_interval_ms == 0U) {
    throw std::invalid_argument("--bitrate-interval must be greater than zero");
  }
  return options;
}

void bind_tensor_output(holoscan::CompileOptions& compile_options,
                        std::string_view operator_path,
                        std::string_view output_port,
                        std::int32_t cuda_device) {
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = operator_path,
      .output_port = output_port,
      .device = holoscan::DeviceId{cuda_device},
  });
}

[[nodiscard]] holoscan::ExecutionPlan build_plan(
    const Options& options, FrameLimit frame_limit) {
  holoscan::Graph graph{"v4l2-userptr-perf"};
  const auto camera = graph.op<V4L2UserPtrSourceOp>(
      "camera", options.device, options.width, options.height, options.fps,
      options.cuda_device, options.sensor_gain, options.exposure_us, std::move(frame_limit));
  const auto convert = graph.op<RawBayerToBgraOp>(
      "convert", options.width, options.height, options.optical_black, options.bayer_grid,
      options.cuda_device, options.raw_shift, options.display_gain, options.gamma);
  constexpr holoscan::TensorRepresentation kDeviceBgra{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = holoscan::ops::kUInt8Dtype,
      .rank = 3U,
  };
  const auto perf = graph.op<PerfOp>("perf", kDeviceBgra, options.bitrate_interval_ms,
                                     options.bitrate_window_size, options.print_cpu_load);
  const auto display = graph.op<X11DisplayOp>(
      "display", options.width, options.height, "Holoscan V4L2 USERPTR Perf");

  const holoscan::ConnectionOptions latest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(camera->frame, convert->input, latest_one);
  graph.add_flow(convert->output, perf->input, latest_one);
  graph.add_flow(perf->output, display->input, latest_one);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions compile_options;
  bind_tensor_output(compile_options, "convert", "output", options.cuda_device);
  bind_tensor_output(compile_options, "perf", "output", options.cuda_device);
  return holoscan::compile(graph, std::move(compile_options));
}

[[nodiscard]] int run(const Options& options) {
  FrameLimit frame_limit{options.frame_limit};
  holoscan::ExecutionPlan plan = build_plan(options, frame_limit);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.dump_plan) {
    std::cout << plan.json() << '\n';
  }
  if (options.validate_only) {
    std::cout << "v4l2_userptr_perf graph validation complete\n";
    return 0;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline = options.duration_seconds == 0U
                            ? std::chrono::steady_clock::time_point::max()
                            : std::chrono::steady_clock::now() +
                                  std::chrono::seconds{options.duration_seconds};
  while (session.is_running() && g_stop_requested == 0 && !frame_limit.reached()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      g_stop_requested = 1;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }
  if (session.is_running()) session.request_stop();
  session.wait();

  const holoscan::RunTermination termination = session.termination();
  if (!session.diagnostics().empty()) {
    std::cerr << "v4l2_userptr_perf runtime diagnostics:\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token << '\n';
    }
  }
  if (termination == holoscan::RunTermination::kFailed ||
      (!session.diagnostics().empty() && termination != holoscan::RunTermination::kStopped)) {
    return 2;
  }
  std::cout << "v4l2_userptr_perf stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "v4l2_userptr_perf");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "v4l2_userptr_perf: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "v4l2_userptr_perf");
    return 64;
  }
}
