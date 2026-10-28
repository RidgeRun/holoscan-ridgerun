// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file interprocess_receiver.cpp
 * @brief Receive and display one catalog-backed Tensor endpoint.
 */

#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/profile.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include "interprocess_src/interprocess_src.hpp"
#include "x11_display/x11_display.hpp"

namespace {

using holoscan::ops::InterprocessEndOfStream;
using holoscan::ops::InterprocessReceiverReady;
using holoscan::ops::InterprocessSrcOp;
using holoscan::ops::X11DisplayOp;

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::filesystem::path catalog;
  std::string endpoint{"video"};
  std::int32_t width{1920};
  std::int32_t height{1080};
  std::int32_t cuda_device{0};
  std::uint32_t poll_milliseconds{5U};
  std::uint32_t duration_seconds{};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output << "Usage: " << program << " --catalog PATH [options]\n\n"
         << "Display one cross-process Tensor endpoint through InterprocessSrc.\n\n"
         << "Options:\n"
         << "  --catalog PATH       Sender endpoint catalog\n"
         << "  --endpoint NAME      Endpoint to receive (default: video)\n"
         << "  --width PIXELS       Display width (default: 1920)\n"
         << "  --height PIXELS      Display height (default: 1080)\n"
         << "  --cuda-device INDEX  CUDA device ordinal (default: 0)\n"
         << "  --poll-ms N          Source polling interval (default: 5)\n"
         << "  --duration SECONDS   Stop after a fixed duration (default: run until stopped)\n"
         << "  --validate           Load the catalog and compile the receiver graph only\n"
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
    } else if (argument == "--endpoint") {
      options.endpoint = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--cuda-device") {
      options.cuda_device =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--poll-ms") {
      options.poll_milliseconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--duration") {
      options.duration_seconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
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
  if (options.endpoint.empty() || options.width <= 0 || options.height <= 0 ||
      options.cuda_device < 0 || options.poll_milliseconds == 0U) {
    throw std::invalid_argument("receiver dimensions, endpoint, and polling period must be valid");
  }
  return options;
}

[[nodiscard]] holoscan::ExecutionPlan build_plan(const Options& options,
                                                 const holoscan::ChannelEndpointCatalog& catalog,
                                                 InterprocessEndOfStream end_of_stream,
                                                 InterprocessReceiverReady receiver_ready) {
  holoscan::Graph graph{"interprocess-receiver"};
  const auto source = graph.op<InterprocessSrcOp>(
      "source", catalog, options.endpoint,
      std::chrono::milliseconds{options.poll_milliseconds}, std::move(end_of_stream),
      std::move(receiver_ready), true);
  const auto display =
      graph.op<X11DisplayOp>("display", options.width, options.height, "Holoscan Interprocess");
  graph.add_flow(source->output, display->input,
                 holoscan::ConnectionOptions{
                     .queue_policy = holoscan::QueuePolicy::kDropOldest,
                     .queue_depth = 1U,
                 });
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
  holoscan::CompileOptions compile_options;
  compile_options.profile = holoscan::Profile::kStandard;
  compile_options.deployment.bind_tensor_output_device(
      holoscan::TensorOutputDevicePlacement{
          .operator_path = "source",
          .output_port = "output",
          .device = holoscan::DeviceId{options.cuda_device},
      });
  return holoscan::compile(graph, std::move(compile_options));
}

[[nodiscard]] int run(const Options& options) {
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(options.catalog);
  if (!catalog) {
    std::cerr << "interprocess_receiver: catalog load failed: "
              << holoscan::runtime_error_message(catalog.error()) << '\n';
    return 1;
  }

  InterprocessEndOfStream end_of_stream;
  InterprocessReceiverReady receiver_ready;
  const holoscan::ExecutionPlan plan =
      build_plan(options, *catalog, end_of_stream, receiver_ready);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate_only) {
    std::cout << "interprocess_receiver graph validation complete\n";
    return 0;
  }

  auto control_runtime = holoscan::ChannelRuntime::create();
  if (!control_runtime) {
    std::cerr << "interprocess_receiver: control channel runtime creation failed: "
              << holoscan::error_code_name(control_runtime.error().code) << '\n';
    return 1;
  }
  auto ready_publisher =
      control_runtime->create_publisher<std::uint32_t>(*catalog, "receiver_ready");
  auto done_subscriber =
      control_runtime->create_subscriber<std::uint32_t>(*catalog, "sender_done");
  if (!ready_publisher || !done_subscriber) {
    std::cerr << "interprocess_receiver: control endpoint creation failed\n";
    return 1;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  holoscan::RunSession session = holoscan::run_async(plan);

  std::cout << "interprocess_receiver: waiting for sender control endpoint\n";
  bool control_failed = false;
  bool readiness_sent = false;
  while (session.is_running() && g_stop_requested == 0 && !readiness_sent) {
    if (!receiver_ready.signaled()) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
      continue;
    }
    auto refreshed = ready_publisher->update_connections();
    if (!refreshed) {
      control_failed = true;
      break;
    }
    auto emitted = ready_publisher->emit(1U);
    if (emitted) {
      readiness_sent = true;
      std::cout << "interprocess_receiver: sender synchronized; receiving frames\n";
      break;
    }
    if (emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured) {
      control_failed = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{5});
  }

  const auto deadline =
      options.duration_seconds == 0U
          ? std::chrono::steady_clock::time_point::max()
          : std::chrono::steady_clock::now() + std::chrono::seconds{options.duration_seconds};
  while (!control_failed && session.is_running() && g_stop_requested == 0 &&
         !end_of_stream.signaled()) {
    auto done = done_subscriber->receive_sample();
    if (done && done->data != 0U) {
      std::cout << "interprocess_receiver: sender completed; stopping receiver\n";
      break;
    }
    if (!done && done.error().code != holoscan::ErrorCode::kNotReady) {
      if (done.error().code != holoscan::ErrorCode::kClosed) {
        control_failed = true;
      }
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }

  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();
  for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
    std::cerr << "interprocess_receiver diagnostic: " << diagnostic.token << '\n';
  }
  if (session.termination() == holoscan::RunTermination::kFailed ||
      !session.diagnostics().empty() || control_failed || !readiness_sent) {
    return 2;
  }
  std::cout << "interprocess_receiver stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "interprocess_receiver");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "interprocess_receiver: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "interprocess_receiver");
    return 64;
  }
}
