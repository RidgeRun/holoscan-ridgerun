// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "interprocess_src/interprocess_src.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <utility>

#include "flow_trace/flow_trace.hpp"
#include "operator_utils/tensor_utils.hpp"

namespace holoscan::ops {

InterprocessSrcOp::InterprocessSrcOp(
    holoscan::ChannelEndpointCatalog catalog, std::string endpoint_name,
    const std::chrono::nanoseconds poll_period, InterprocessEndOfStream end_of_stream,
    InterprocessReceiverReady receiver_ready, const bool retain_imported_tensor)
    : catalog_(std::move(catalog)),
      endpoint_name_(std::move(endpoint_name)),
      poll_period_(poll_period),
      end_of_stream_(std::move(end_of_stream)),
      receiver_ready_(std::move(receiver_ready)),
      retain_imported_tensor_(retain_imported_tensor) {
  if (endpoint_name_.empty()) {
    throw std::invalid_argument("InterprocessSrcOp endpoint name must not be empty");
  }
  if (poll_period_ <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("InterprocessSrcOp poll period must be positive");
  }
  auto endpoint = catalog_.endpoint(endpoint_name_);
  if (!endpoint) {
    throw std::invalid_argument("InterprocessSrcOp endpoint is missing from the catalog: " +
                                endpoint_name_);
  }
  const auto requirements = endpoint->tensor_requirements();
  if (!requirements || !requirements->max_rank() || !requirements->max_byte_span() ||
      *requirements->max_rank() == 0U || *requirements->max_byte_span() == 0U) {
    throw std::invalid_argument("InterprocessSrcOp endpoint has no bounded Tensor contract: " +
                                endpoint_name_);
  }
  output_max_rank_ = *requirements->max_rank();
  output_max_byte_span_ = *requirements->max_byte_span();
}

void InterprocessSrcOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {
              .memory_kind = holoscan::MemoryKind::kCudaDevice,
              .dtype = DLDataType{kDLUInt, 8U, 1U},
              .rank = 3U,
          },
          .bounds = holoscan::TensorBounds{
              .max_rank = output_max_rank_,
              .max_byte_span = output_max_byte_span_,
          },
          // Keep a Runtime pool declaration even in retained-import mode. The compiler uses this
          // storage obligation and its device binding to materialize the CUDA stream required by
          // downstream Tensor access. Retained-import compute does not allocate from or write to
          // this pool; it emits the imported Tensor handle directly.
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
}

holoscan::Contract InterprocessSrcOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnClock{.period = poll_period_});
  return result;
}

void InterprocessSrcOp::start() {
  const cudaError_t cuda_status = cudaFree(nullptr);
  if (cuda_status != cudaSuccess) {
    throw holoscan::RuntimeError(holoscan::Error{
        holoscan::ErrorCode::kNotSupported,
        std::string{"InterprocessSrcOp could not initialize CUDA: "} +
            cudaGetErrorString(cuda_status)});
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    throw holoscan::RuntimeError(runtime.error());
  }
  runtime_ = std::make_unique<holoscan::ChannelRuntime>(std::move(*runtime));
  reader_ = std::make_unique<InterprocessReader>(*runtime_, catalog_, endpoint_name_);
  auto started = reader_->start();
  if (!started) {
    throw holoscan::RuntimeError(started.error());
  }
  receiver_ready_.signal();
}

void InterprocessSrcOp::stop() {
  if (reader_) {
    reader_->stop();
  }
  reader_.reset();
  runtime_.reset();
}

holoscan::expected<void, holoscan::Error> InterprocessSrcOp::compute(
    holoscan::ExecutionContext& context) {
  auto received = reader_->receive();
  if (!received) {
    if (received.error().code == holoscan::ErrorCode::kClosed) {
      end_of_stream_.signal();
      return {};
    }
    return holoscan::make_unexpected(std::move(received).error());
  }
  if (!received->has_value()) {
    return {};
  }

  auto sample = std::move(received->value());
  const std::uint64_t frame_id = sample.metadata.frame_id;
  const std::int64_t trace_receive_us =
      holoscan::ops::FlowTrace::now_microseconds();
  const holoscan::Tensor& input = sample.data;
  const std::int64_t input_bytes = input.nbytes();
  if (input.data() == nullptr || input.device().device_type != kDLCUDA ||
      !holoscan::ops::same_dtype(
          input.dtype(), DLDataType{kDLUInt, 8U, 1U}) ||
      input.shape_span().size() != 3U || !input.is_contiguous() || input_bytes <= 0 ||
      static_cast<std::uint64_t>(input_bytes) > output_max_byte_span_) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kInvalidArgument,
        "InterprocessSrcOp expected a bounded contiguous CUDA uint8 rank-3 Tensor"});
  }

  if (retain_imported_tensor_) {
    // The imported Tensor owns the HoloMQ read claim. Passing the complete Sample through keeps
    // that claim alive until the runtime and every downstream consumer release their handles.
    auto emitted = output.emit(std::move(sample));
    if (!emitted) {
      return emitted;
    }
    holoscan::ops::FlowTrace::instance().record(
        frame_id,
        "interprocess_source",
        trace_receive_us,
        holoscan::ops::FlowTrace::now_microseconds());
    return {};
  }

  auto loan = output.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = input.shape_span(), .dtype = input.dtype()});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write(context.cuda_stream());
  if (!writer) {
    return holoscan::make_unexpected(std::move(writer).error());
  }
  if (writer->data() == nullptr ||
      writer->byte_size() != static_cast<std::size_t>(input_bytes)) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        "InterprocessSrcOp received an invalid local Tensor allocation"});
  }

  const cudaStream_t stream = context.cuda_stream().get();
  cudaError_t status = cudaMemcpyAsync(writer->data(),
                                       input.data(),
                                       static_cast<std::size_t>(input_bytes),
                                       cudaMemcpyDeviceToDevice,
                                       stream);
  if (status == cudaSuccess) {
    // The external subscriber's read claim ends when this compute call returns.
    // Complete the copy before releasing that imported IPC allocation.
    status = cudaStreamSynchronize(stream);
  }
  if (status != cudaSuccess) {
    return holoscan::make_unexpected(holoscan::Error{
        holoscan::ErrorCode::kFailure,
        std::string{"InterprocessSrcOp CUDA bridge copy failed: "} + cudaGetErrorString(status)});
  }
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }
  auto emitted = output.emit(
      std::move(*loan),
      holoscan::ops::forwarded_emit_options(sample.metadata));
  if (!emitted) {
    return emitted;
  }
  holoscan::ops::FlowTrace::instance().record(
      frame_id,
      "interprocess_source",
      trace_receive_us,
      holoscan::ops::FlowTrace::now_microseconds());
  return {};
}

}  // namespace holoscan::ops
