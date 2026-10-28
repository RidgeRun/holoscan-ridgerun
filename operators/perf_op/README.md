# PerfOp

`PerfOp` is a reusable Holoscan SDK (HSDK) 5 Tensor identity operator inspired by
[RidgeRun gst-perf](https://github.com/RidgeRun/gst-perf). It can be inserted at
any Tensor edge to report instantaneous and average framerate and bitrate, plus
optional aggregate system CPU load.

The operator is independent of camera, image format, CUDA, inference, and
display operators. Its ports deliberately leave Tensor dtype, rank, layout, and
memory kind unconstrained.

## Data behavior

`compute()` receives and moves the complete input `Sample<Tensor>` to the output.
This preserves the original Tensor storage and its capture, trace, frame, and
semantic metadata. It performs no payload allocation, payload read, CUDA call,
CUDA synchronization, or frame copy.

Per sample, it only reads `Tensor::nbytes()` and updates relaxed atomic counters.
A separate host thread performs periodic calculations and logging, including
intervals in which no samples arrive.

The bitrate is the logical Tensor byte throughput at the insertion point. It is
not an encoded-video bitrate unless the Tensor itself contains encoded data.

## C++ integration

Declare the operator dependency in the application registration:

```cmake
add_holohub_application(my_application
    DEPENDS OPERATORS perf_op)
```

Link it from the application target:

```cmake
target_link_libraries(my_application PRIVATE holoscan::perf_op)
```

Insert it in a graph:

```cpp
#include "perf_op/perf_op.hpp"

const holoscan::TensorRepresentation representation{
    .memory_kind = holoscan::MemoryKind::kCudaDevice,
    .dtype = uint8_dtype,
    .rank = 3U,
};
const auto perf = graph.op<holoscan::ops::PerfOp>(
    "perf",
    representation,  // optional but recommended in HSDK 5
    1000U,  // reporting/bitrate interval in milliseconds
    5U,     // bitrate moving-average window; zero means all reports
    true);  // include aggregate CPU load

graph.add_flow(producer->output, perf->input, connection_options);
graph.add_flow(perf->output, consumer->input, connection_options);
```

Constructor arguments correspond to gst-perf's `bitrate-interval`,
`bitrate-window-size`, and `print-cpu-load` properties.

The representation argument is configurable rather than tied to BGRA. Supplying
the representation is recommended in HSDK 5 so the compiler can materialize the
downstream runtime binding. With a representation, `PerfOp` derives the maximum
slot size from the connected producer through `input_max_byte_span(input)` and
declares a runtime pool. HSDK 5 needs that physical declaration for the output and
CUDA-stream binding, but `compute()` forwards the input sample rather than
allocating from or copying into the pool. The application must bind the PerfOp
output to the appropriate CUDA device when the representation is CUDA memory.

## Scope

Holoscan ports are strongly typed, so this version is universal for
`holoscan::Tensor` pipelines rather than arbitrary C++ payload types. Supporting
other payload types requires a typed adapter for each payload because only a
Tensor provides a generic logical byte count through `nbytes()`.

The payload path is zero-copy. Inserting any operator still adds a scheduler
activation and graph queue boundary; therefore it cannot promise literally zero
scheduling latency. A depth-one `DropOldest` connection is recommended for live
pipelines that prioritize the newest frame.
