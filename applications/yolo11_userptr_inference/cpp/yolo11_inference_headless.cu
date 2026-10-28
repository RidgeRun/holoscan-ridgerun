// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file yolo11_inference_headless.cu
 * @brief Minimal IMX274 capture and YOLO11 inference pipeline for Holoscan SDK (HSDK) 5.x.
 *
 * V4L2 USERPTR RG10 -> fused Bayer/letterbox/normalize -> TensorRT -> Fakesink
 */

#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
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
#include "raw_bayer_to_yolo_tensor.hpp"
#include "fakesink_op/fakesink_op.hpp"
#include "tensorrt_inference/tensorrt_inference.hpp"
#include "v4l2_userptr_source/v4l2_userptr_source.hpp"

namespace {

using holoscan::ops::FrameLimit;
using holoscan::ops::FakesinkOp;
using holoscan::ops::TensorRtInferenceOp;
using holoscan::ops::V4L2UserPtrSourceOp;
using holoscan::ops::RawBayerToYoloTensorOp;

constexpr std::string_view kDeepStreamEngine{
    "/home/nvidia/DeepStream-Yolo/yolo11n_user.onnx_b1_gpu0_fp16.engine"};
constexpr std::int32_t kNetworkWidth = 640;
constexpr std::int32_t kNetworkHeight = 640;
constexpr std::size_t kMaximumInferenceOutputBytes = 16U * 1024U * 1024U;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::string device{"/dev/video0"};
  std::string model{kDeepStreamEngine};
  std::string engine_cache;
  std::int32_t width{1920};
  std::int32_t height{1080};
  std::int32_t fps{60};
  std::uint32_t duration_seconds{};
  std::uint64_t frame_limit{};
  std::int32_t cuda_device{0};
  std::int32_t optical_black{0};
  std::int32_t raw_shift{6};
  std::int32_t bayer_grid{1};
  float input_gain{8.0F};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n\n"
      << "Headless V4L2 USERPTR IMX274 capture and YOLO11 TensorRT inference.\n\n"
      << "Options:\n"
      << "  --device PATH          V4L2 capture device (default: /dev/video0)\n"
      << "  --model PATH           DeepStream-Yolo ONNX or TensorRT engine\n"
      << "                         (default: " << kDeepStreamEngine << ")\n"
      << "  --engine-cache PATH    Cache path when --model is ONNX\n"
      << "  --width PIXELS         RG10 capture width (default: 1920)\n"
      << "  --height PIXELS        RG10 capture height (default: 1080)\n"
      << "  --fps RATE             Capture rate (default: 60)\n"
      << "  --duration SECONDS     Stop after a duration (default: unlimited)\n"
      << "  --frame-limit COUNT    Stop after captured frames (default: unlimited)\n"
      << "  --cuda-device INDEX    CUDA device ordinal (default: 0)\n"
      << "  --optical-black N      RAW10 black-level offset (default: 0)\n"
      << "  --raw-shift N          Right shift for RAW10 samples (default: 6)\n"
      << "  --input-gain X         Linear gain before YOLO normalization (default: 8.0)\n"
      << "  --bayer-grid N         0=BGGR, 1=RGGB, 2=GRBG, 3=GBRG (default: 1)\n"
      << "  --validate             Compile graph without camera/model\n"
      << "  -h, --help             Show this help text\n";
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] float parse_float(std::string_view text, std::string_view option) {
  float value{};
  const char* begin = text.data();
  const char* end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
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
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--model") {
      options.model = option_value(index, argc, argv, argument);
    } else if (argument == "--engine-cache") {
      options.engine_cache = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps = parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
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
    } else if (argument == "--input-gain") {
      options.input_gain = parse_float(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--bayer-grid") {
      options.bayer_grid =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.device.empty() || options.model.empty()) {
    throw std::invalid_argument("--device and --model must not be empty");
  }
  if (options.width <= 0 || options.height <= 0 || options.fps <= 0 ||
      options.cuda_device < 0) {
    throw std::invalid_argument("dimensions and rate must be positive; CUDA device must be nonnegative");
  }
  if ((options.width & 1) != 0 || (options.height & 1) != 0) {
    throw std::invalid_argument("--width and --height must be even for Bayer capture");
  }
  if (options.optical_black < 0 || options.optical_black >= 1023 ||
      options.raw_shift < 0 || options.raw_shift > 6 ||
      options.bayer_grid < 0 || options.bayer_grid > 3 ||
      !std::isfinite(options.input_gain) || options.input_gain <= 0.0F) {
    throw std::invalid_argument("invalid RAW-to-model preprocessing option");
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

[[nodiscard]] holoscan::ExecutionPlan build_plan(const Options& options, FrameLimit frame_limit) {
  holoscan::Graph graph{"yolo11-userptr-inference-headless"};
  const auto camera = graph.op<V4L2UserPtrSourceOp>(
      "camera", options.device, options.width, options.height, options.fps, options.cuda_device, 8000000, 15649, std::move(frame_limit));
  const auto preprocess = graph.op<RawBayerToYoloTensorOp>(
      "preprocess", options.width, options.height, kNetworkWidth, kNetworkHeight,
      options.optical_black, options.bayer_grid, options.raw_shift, options.input_gain);
  const auto inference = graph.op<TensorRtInferenceOp>(
      "inference", options.model, options.engine_cache, options.cuda_device,
      kMaximumInferenceOutputBytes);
  const auto sink = graph.op<FakesinkOp>("fakesink");

  const holoscan::ConnectionOptions latest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };
  graph.add_flow(camera->frame, preprocess->input, latest_one);
  graph.add_flow(preprocess->output, inference->input, latest_one);
  graph.add_flow(inference->output, sink->input, latest_one);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  holoscan::CompileOptions compile_options;
  bind_tensor_output(compile_options, "preprocess", "output", options.cuda_device);
  bind_tensor_output(compile_options, "inference", "output", options.cuda_device);
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
    std::cout << "yolo11_inference headless graph validation complete\n";
    return 0;
  }
  if (!std::filesystem::is_regular_file(options.model)) {
    std::cerr << "YOLO11 model/engine was not found: " << options.model << '\n';
    return 66;
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
    std::cerr << "yolo11_inference runtime diagnostics:\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token;
      if (!diagnostic.required.empty()) std::cerr << " required=" << diagnostic.required;
      if (!diagnostic.provided.empty()) std::cerr << " provided=" << diagnostic.provided;
      std::cerr << '\n';
    }
  }
  if (termination == holoscan::RunTermination::kFailed ||
      (!session.diagnostics().empty() && termination != holoscan::RunTermination::kStopped)) {
    return 2;
  }
  std::cout << "yolo11_inference stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "yolo11_inference");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "yolo11_inference: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "yolo11_inference");
    return 64;
  }
}
