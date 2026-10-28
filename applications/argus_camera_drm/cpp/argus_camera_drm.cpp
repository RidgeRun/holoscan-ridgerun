// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <charconv>
#include <chrono>
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

#include "argus_camera/argus_camera.hpp"
#include "drm_bgra_display/drm_bgra_display.hpp"

namespace {

using holoscan::ops::FrameLimit;
using holoscan::ops::ArgusCameraOp;
using holoscan::ops::DrmBgraDisplayOp;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::uint32_t camera_index{};
  std::uint32_t sensor_mode_index{};
  std::int32_t width{};
  std::int32_t height{};
  std::int32_t fps{30};
  std::uint32_t duration_seconds{};
  std::uint64_t frame_limit{};
  std::string drm_device{"/dev/dri/card2"};
  std::uint32_t connector_id{};
  std::string drm_mode{};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " --width PIXELS --height PIXELS [options]\n\n"
      << "Direct Jetson MIPI CSI capture through Argus, displayed through exclusive DRM/KMS.\n\n"
      << "Options:\n"
      << "  --camera INDEX       Argus camera device index (default: 0)\n"
      << "  --sensor-mode INDEX  Argus sensor mode index (default: 0)\n"
      << "  --width PIXELS       Exact selected sensor-mode width (required)\n"
      << "  --height PIXELS      Exact selected sensor-mode height (required)\n"
      << "  --fps RATE           Capture rate supported by the mode (default: 30)\n"
      << "  --duration SECONDS   Stop after this duration (default: until stopped)\n"
      << "  --frame-limit COUNT   Stop after this many captured frames (default: unlimited)\n"
      << "  --drm-device PATH    DRM card (default: /dev/dri/card2)\n"
      << "  --connector ID       DRM connector ID (default: auto-select)\n"
      << "  --mode WIDTHxHEIGHT  DRM mode (default: capture dimensions)\n"
      << "  --validate           Compile the graph without opening camera/display\n"
      << "  -h, --help           Show this help text\n\n"
      << "Press Ctrl-C to stop the exclusive DRM run.\n";
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
    } else if (argument == "--camera") {
      options.camera_index =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--sensor-mode") {
      options.sensor_mode_index =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--width") {
      options.width =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--duration") {
      options.duration_seconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--frame-limit") {
      options.frame_limit =
          parse_integer<std::uint64_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--drm-device") {
      options.drm_device = option_value(index, argc, argv, argument);
    } else if (argument == "--connector") {
      options.connector_id =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--mode") {
      options.drm_mode = option_value(index, argc, argv, argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (!options.show_help &&
      (options.width <= 0 || options.height <= 0 || options.fps <= 0)) {
    throw std::invalid_argument("--width, --height, and --fps must be positive");
  }
  if (!options.show_help && ((options.width & 1) != 0 || (options.height & 1) != 0)) {
    throw std::invalid_argument("--width and --height must be even for Argus NV12");
  }
  if (!options.show_help && options.drm_device.empty()) {
    throw std::invalid_argument("--drm-device must not be empty");
  }
  if (!options.show_help && options.drm_mode.empty()) {
    options.drm_mode = std::to_string(options.width) + "x" + std::to_string(options.height);
  }
  return options;
}

[[nodiscard]] holoscan::ExecutionPlan build_plan(
    const Options& options, FrameLimit frame_limit) {
  holoscan::Graph graph{"argus-camera-drm"};
  const auto camera = graph.op<ArgusCameraOp>("camera",
                                               options.camera_index,
                                               options.sensor_mode_index,
                                               options.width,
                                               options.height,
                                               options.fps,
                                               std::move(frame_limit));
  const auto display = graph.op<DrmBgraDisplayOp>("display",
                                                  options.width,
                                                  options.height,
                                                  options.drm_device,
                                                  options.connector_id,
                                                  options.drm_mode);
  graph.add_flow(camera->frame,
                 display->input,
                 holoscan::ConnectionOptions{
                     .queue_policy = holoscan::QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "camera",
          .output_port = "frame",
          .device = holoscan::DeviceId{0},
      });
  return holoscan::compile(graph, std::move(compile_options));
}

[[nodiscard]] int run(const Options& options) {
  FrameLimit frame_limit{options.frame_limit};
  holoscan::ExecutionPlan plan = build_plan(options, frame_limit);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate_only) {
    std::cout << "argus_camera_drm graph validation complete\n";
    return 0;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      options.duration_seconds == 0U
          ? std::chrono::steady_clock::time_point::max()
          : std::chrono::steady_clock::now() +
                std::chrono::seconds{options.duration_seconds};
  while (session.is_running() && g_stop_requested == 0 &&
         !frame_limit.reached()) {
    if (std::chrono::steady_clock::now() >= deadline) {
      g_stop_requested = 1;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();

  if (!session.diagnostics().empty()) {
    std::cerr << "argus_camera_drm runtime diagnostics:\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token;
      if (!diagnostic.required.empty()) {
        std::cerr << " required=" << diagnostic.required;
      }
      if (!diagnostic.provided.empty()) {
        std::cerr << " provided=" << diagnostic.provided;
      }
      std::cerr << '\n';
    }
  }
  if (session.termination() == holoscan::RunTermination::kFailed) {
    return 2;
  }
  std::cout << "argus_camera_drm stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "argus_camera_drm");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "argus_camera_drm: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "argus_camera_drm");
    return 64;
  }
}
