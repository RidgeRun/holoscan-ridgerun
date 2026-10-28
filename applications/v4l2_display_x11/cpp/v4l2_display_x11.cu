// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file v4l2_display_x11.cu
 * @brief Minimal V4L2-to-X11 display graph without inference.
 *
 *   V4L2 RG10 -> Bayer-to-BGRA -> X11
 *
 * This is the window-manager path used as the non-exclusive baseline. The
 * MMAP and CUDA-mapped USERPTR capture paths produce uint16 RG10 tensors that
 * RawBayerToBgraOp converts to BGRA on CUDA for X11 presentation.
 */

#include <array>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "raw_bayer_to_bgra/raw_bayer_to_bgra.hpp"
#include "v4l2_source/v4l2_source.hpp"
#include "v4l2_userptr_source/v4l2_userptr_source.hpp"
#include "x11_display/x11_display.hpp"

namespace {

using holoscan::ops::RawBayerToBgraOp;
using holoscan::ops::V4L2SourceOp;
using holoscan::ops::V4L2UserPtrSourceOp;
using holoscan::ops::X11DisplayOp;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::string device{"/dev/video0"};
  int width{1920};
  int height{1080};
  int fps{60};
  int cuda_device{0};
  int optical_black{50};
  int raw_shift{6};
  float display_gain{12.0F};
  float gamma{2.2F};
  int bayer_grid{1};
  std::string capture_memory{"mmap"};
  std::string title{"Holoscan V4L2 Display X11"};
  unsigned int duration_seconds{};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

float parse_float(std::string_view text, std::string_view option) {
  float value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires a number");
  }
  return value;
}

std::string option_value(int& index, int argc, char** argv, std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[++index];
}

void print_usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " [options]\n\n"
         << "HSDK 5 V4L2 RG10 capture, CUDA Bayer processing, and X11 display.\n\n"
         << "Options:\n"
         << "  --device PATH         V4L2 device (default: /dev/video0)\n"
         << "  --width PIXELS        Exact RG10 capture width (default: 1920)\n"
         << "  --height PIXELS       Exact RG10 capture height (default: 1080)\n"
         << "  --fps RATE            Exact capture rate (default: 60)\n"
         << "  --cuda-device INDEX   CUDA device (default: 0)\n"
         << "  --capture-memory MODE Capture memory: mmap or userptr (default: mmap)\n"
         << "  --optical-black N     RAW10 black-level offset (default: 50)\n"
         << "  --raw-shift N         Right shift for RAW10 samples (default: 6)\n"
         << "  --display-gain X      Digital gain after black level (default: 12.0)\n"
         << "  --gamma X             Display gamma encoding (default: 2.2)\n"
         << "  --bayer-grid N        0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG (default: 1)\n"
         << "  --title TEXT          X11 window title\n"
         << "  --duration SECONDS    Stop after a fixed duration\n"
         << "  --validate             Compile only; do not open camera or X11\n"
         << "  -h, --help             Show this help text\n\n"
         << "Press q, Escape, or Ctrl-C to stop a live run.\n";
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "-h" || argument == "--help") {
      options.show_help = true;
    } else if (argument == "--validate") {
      options.validate_only = true;
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--cuda-device") {
      options.cuda_device = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--capture-memory") {
      options.capture_memory = option_value(index, argc, argv, argument);
    } else if (argument == "--optical-black") {
      options.optical_black =
          parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--raw-shift") {
      options.raw_shift = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--display-gain") {
      options.display_gain = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--gamma") {
      options.gamma = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--bayer-grid") {
      options.bayer_grid = parse_integer<int>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--title") {
      options.title = option_value(index, argc, argv, argument);
    } else if (argument == "--duration") {
      options.duration_seconds = parse_integer<unsigned int>(
          option_value(index, argc, argv, argument), argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (options.device.empty() || options.title.empty() || options.width <= 0 ||
      options.height <= 0 || options.fps <= 0 || options.cuda_device < 0 ||
      (options.capture_memory != "mmap" && options.capture_memory != "userptr") ||
      (options.width & 1) != 0 || (options.height & 1) != 0) {
    throw std::invalid_argument(
        "device/title must be nonempty; width, height, fps, and cuda-device must be valid; "
        "width and height must be even; capture-memory must be mmap or userptr");
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
  if (options.bayer_grid < 0 || options.bayer_grid > 3) {
    throw std::invalid_argument("--bayer-grid must be in [0,3]");
  }
  return options;
}

holoscan::ExecutionPlan build_plan(const Options& options) {
  holoscan::Graph graph{"v4l2-display-x11"};
  const auto convert = graph.op<RawBayerToBgraOp>("convert",
                                                   options.width,
                                                   options.height,
                                                   options.optical_black,
                                                   options.bayer_grid,
                                                   options.cuda_device,
                                                   options.raw_shift,
                                                   options.display_gain,
                                                   options.gamma);
  const auto display = graph.op<X11DisplayOp>(
      "display", options.width, options.height, options.title);

  const holoscan::ConnectionOptions latest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  if (options.capture_memory == "userptr") {
    const auto camera = graph.op<V4L2UserPtrSourceOp>(
        "camera", options.device, options.width, options.height, options.fps, options.cuda_device);
    graph.add_flow(camera->frame, convert->input, latest_one);
  } else {
    const auto camera = graph.op<V4L2SourceOp>(
        "camera", options.device, options.width, options.height, options.fps);
    graph.add_flow(camera->frame, convert->input, latest_one);
  }
  graph.add_flow(convert->output, display->input, latest_one);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions compile_options;
  compile_options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "camera",
          .output_port = "frame",
          .device = holoscan::DeviceId{options.cuda_device},
      });
  compile_options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "convert",
          .output_port = "output",
          .device = holoscan::DeviceId{options.cuda_device},
      });
  return holoscan::compile(graph, std::move(compile_options));
}

int run(const Options& options) {
  holoscan::ExecutionPlan plan = build_plan(options);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate_only) {
    std::cout << "v4l2_display_x11 graph validation complete\n";
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
  while (session.is_running() && g_stop_requested == 0) {
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
    std::cerr << "v4l2_display_x11 runtime diagnostics:\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token << '\n';
    }
  }
  if (session.termination() == holoscan::RunTermination::kFailed ||
      (!session.diagnostics().empty() &&
       session.termination() != holoscan::RunTermination::kStopped)) {
    return 2;
  }
  std::cout << "v4l2_display_x11 stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "v4l2_display_x11");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "v4l2_display_x11: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "v4l2_display_x11");
    return 64;
  }
}
