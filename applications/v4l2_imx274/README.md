# IMX274 V4L2 capture to X11

This Holoscan SDK (HSDK) 5 C++ application captures an IMX274 RG10 Bayer stream
through V4L2 MMAP, processes it on CUDA, and displays the resulting BGRA image
in an X11 window.

## Pipeline

```text
V4L2SourceOp (RG10 MMAP) -> RawBayerToBgraOp -> X11DisplayOp
```

`V4L2SourceOp` copies each dequeued MMAP frame into a plan-owned CUDA tensor.
`RawBayerToBgraOp` performs Bayer reconstruction, black-level correction,
digital gain, gamma encoding, and BGRA generation on CUDA. `X11DisplayOp`
copies the final image to host memory for Xlib presentation.

Connections use a queue depth of one with drop-oldest behavior.

## Prerequisites

The camera node must expose single-plane RG10 at the exact requested dimensions
and frame rate:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
export DISPLAY=:0
```

For a native build, install `v4l-utils` and the X11 development package.

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_imx274 live --local --language cpp
./holoscan_ridgerun run v4l2_imx274 validate \
  --local --language cpp --no-local-build
```

## Run

```bash
./holoscan_ridgerun run v4l2_imx274 live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

The Bayer grid values are `0=BGGR`, `1=RGGB`, `2=GRBG`, and `3=GBRG`.
Adjust `--optical-black`, `--display-gain`, and `--gamma` for the sensor
and lighting conditions. Press `q`, Escape, close the window, or press Ctrl-C
to stop an unlimited run.

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
