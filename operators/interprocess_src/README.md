# InterprocessSrcOp

`InterprocessSrcOp` is a reusable Holoscan Tensor source that receives samples
from one catalog-backed HoloMQ endpoint.

The operator is clock-triggered and polls the subscriber at the configured
period. A poll with no available sample emits nothing. A closed channel signals
the shared `InterprocessEndOfStream` state so the owning application can stop its
run session cleanly.

The endpoint must exist in the catalog and declare non-zero Tensor rank and
byte-span bounds. The current output contract accepts bounded, contiguous CUDA
`uint8` rank-3 Tensors.

Two ownership modes are available:

- `retain_imported_tensor=false`: allocate from the local runtime pool and
  make one CUDA device-to-device copy before releasing the HoloMQ claim.
- `retain_imported_tensor=true`: move the complete imported sample to the
  output so its Tensor owns the HoloMQ read claim through downstream queues.

## C++ integration

```cpp
#include "interprocess_src/interprocess_src.hpp"

using holoscan::ops::InterprocessEndOfStream;
using holoscan::ops::InterprocessReceiverReady;
using holoscan::ops::InterprocessSrcOp;

InterprocessEndOfStream end_of_stream;
InterprocessReceiverReady receiver_ready;

const auto source = graph.op<InterprocessSrcOp>(
    "source",
    catalog,
    "video",
    std::chrono::milliseconds{5},
    end_of_stream,
    receiver_ready,
    true);
```

Bind the source output to the CUDA device during compilation:

```cpp
options.deployment.bind_tensor_output_device(
    holoscan::TensorOutputDevicePlacement{
        .operator_path = "source",
        .output_port = "output",
        .device = holoscan::DeviceId{cuda_device},
    });
```

The sender and its exported allocation must remain alive while retained Tensor
claims are in use. A depth-one `DropOldest` connection is recommended for live
display pipelines.
