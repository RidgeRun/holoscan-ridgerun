# V4L2 USERPTR performance monitor

This Holoscan SDK (HSDK) 5 application extends `v4l2_userptr_x11` with a lightweight
performance identity operator:

```text
V4L2 USERPTR RG10 -> CUDA Bayer-to-BGRA -> PerfOp -> X11DisplayOp
```

`PerfOp` forwards the original `Sample<Tensor>` by move. It does not allocate a
frame, read the CUDA payload, synchronize CUDA, or perform a payload copy. Its
per-frame work is one metadata read and two relaxed atomic additions. A separate
reporter thread prints instantaneous and mean FPS and bitrate; optional CPU load
is the normalized aggregate system utilization obtained from `/proc/stat`.

The reported bitrate is the logical BGRA tensor throughput at the operator, not
an encoded-stream bitrate.

## Build and validate

```bash
./holoscan_ridgerun build v4l2_userptr_perf live --local --language cpp
./holoscan_ridgerun run v4l2_userptr_perf validate \
  --local --language cpp --no-local-build
```

Validation compiles the graph without opening the camera or X11 window.

## Run

```bash
./holoscan_ridgerun run v4l2_userptr_perf live \
  --local \
  --language cpp \
  --no-local-build \
  --run-args='--frame-limit 600 --bitrate-interval 1000 --bitrate-window-size 5 --print-cpu-load'
```

HSDK 5 requires a physical resource for each `Output<Tensor>::emit()` binding.
`PerfOp` therefore declares a runtime-pool capacity derived at compile time from
its connected input and the application binds that output to the same CUDA
device. The original input sample is still forwarded directly; the pool exists
for HSDK 5 port/stream materialization and is not used to copy or rebuild frames.
Depth-one `DropOldest` queues prevent backlog and prioritize the newest frame.
