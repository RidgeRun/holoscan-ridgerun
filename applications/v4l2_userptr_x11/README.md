# IMX274 V4L2 USERPTR capture to X11

This Holoscan SDK (HSDK) 5 C++ application captures an IMX274 RG10 Bayer stream
into CUDA-mapped V4L2 USERPTR buffers, processes it on CUDA, and displays the
resulting BGRA image in an X11 window.

## Pipeline

```text
V4L2UserPtrSourceOp -> RawBayerToBgraOp -> X11DisplayOp
```

The capture slots are pinned host allocations queued to V4L2 as USERPTR and
mapped into the CUDA address space. The camera and downstream CUDA operator use
the same backing allocation, so the capture operator does not perform an
explicit host-to-device copy. USERPTR support depends on the V4L2 driver.

`RawBayerToBgraOp` generates the BGRA tensor on CUDA. `X11DisplayOp` performs
the device-to-host copy required for Xlib presentation. Connections use a queue
depth of one with drop-oldest behavior.

## Prerequisites

Verify that the camera supports RG10 and V4L2 USERPTR for the selected mode:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
export DISPLAY=:0
```

For a native build, install `v4l-utils` and `libx11-dev`.

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_userptr_x11 live --local --language cpp
./holoscan_ridgerun run v4l2_userptr_x11 validate \
  --local --language cpp --no-local-build
```

## Run

```bash
./holoscan_ridgerun run v4l2_userptr_x11 live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

Use `--sensor-gain` and `--exposure-us` to control the Tegra camera settings.
The Bayer grid values are `0=BGGR`, `1=RGGB`, `2=GRBG`, and `3=GBRG`.
Press `q`, Escape, close the window, or press Ctrl-C to stop an unlimited run.

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
