// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "fakesink_op/fakesink_op.hpp"
#include "flow_trace/flow_trace.hpp"

#include <cstddef>
#include <iostream>
#include <utility>

namespace holoscan::ops {

void FakesinkOp::setup(holoscan::OperatorSpec& spec) {
  spec.input(input, "input").queue_depth(1U);
}

holoscan::Contract FakesinkOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

holoscan::expected<void, holoscan::Error> FakesinkOp::compute(
    holoscan::ExecutionContext&) {
  auto received = input.receive();
  if (!received) {
    return holoscan::make_unexpected(std::move(received).error());
  }

  const std::uint64_t trace_frame_id = received->metadata.frame_id;
  const std::int64_t trace_receive_us =
      holoscan::ops::FlowTrace::now_microseconds();
  if (verbose_) {
    const holoscan::Tensor& tensor = received->data;
    const auto shape = tensor.shape_span();
    const auto dtype = tensor.dtype();
    const auto device = tensor.device();

    std::cout << "fakesink: buffer=" << (++buffers_received_) << " shape=[";
    for (std::size_t index = 0; index < shape.size(); ++index) {
      if (index != 0U) {
        std::cout << ',';
      }
      std::cout << shape[index];
    }
    std::cout << "] dtype=" << static_cast<int>(dtype.code) << ':'
              << static_cast<int>(dtype.bits) << ':' << static_cast<int>(dtype.lanes)
              << " device=" << static_cast<int>(device.device_type) << ':' << device.device_id
              << " bytes=" << tensor.nbytes()
              << " sequence=" << received->metadata.sequence
              << " frame_id=" << received->metadata.frame_id << '\n' << std::flush;
  }

  // Destruction of the received sample releases its plan-owned storage. The
  // payload remains on-device: no synchronization, host copy, or decode occurs.
  holoscan::ops::FlowTrace::instance().record(
      trace_frame_id, "sink", trace_receive_us,
      holoscan::ops::FlowTrace::now_microseconds());
  return {};
}

}  // namespace holoscan::ops
