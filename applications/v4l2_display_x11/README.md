# V4L2 RG10 to X11

This Holoscan SDK (HSDK) 5 C++ application captures RG10 Bayer frames from an
IMX274-compatible V4L2 device, converts them to BGRA on CUDA, and displays them
in an X11 window. It supports MMAP and CUDA-mapped USERPTR capture paths.

## Pipeline

```text
V4L2SourceOp (MMAP) or V4L2UserPtrSourceOp (USERPTR)
  -> RawBayerToBgraOp
  -> X11DisplayOp
```

Both capture paths emit a CUDA-visible `uint16 [height,width,1]` RG10 tensor.
`RawBayerToBgraOp` performs Bayer demosaicing, black-level correction, digital
gain, gamma encoding, and BGRA conversion on CUDA. `X11DisplayOp` performs the
device-to-host copy required for Xlib presentation.

All connections use a queue depth of one with drop-oldest behavior.

## Prerequisites

Confirm that the camera exposes the expected RG10 mode:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

For a native build, install the V4L2 and X11 dependencies and select an
accessible display:

```bash
sudo apt update
sudo apt install v4l-utils libx11-dev
export DISPLAY=:0
```

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_display_x11 live --local --language cpp
./holoscan_ridgerun run v4l2_display_x11 validate \
  --local --language cpp --no-local-build
```

Validation compiles the graph without opening the camera or X11 window.

## Run

```bash
./holoscan_ridgerun run v4l2_display_x11 live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --capture-memory mmap --width 1920 --height 1080 --fps 60 --optical-black 50 --raw-shift 6 --display-gain 12.0 --gamma 2.2 --bayer-grid 1 --title "Holoscan V4L2 RG10" --duration 30'
```

Use `--capture-memory userptr` to select CUDA-mapped USERPTR capture. Adjust
`--optical-black`, `--display-gain`, and `--gamma` for the sensor and lighting.
The Bayer-grid values are `0=BGGR`, `1=RGGB`, `2=GRBG`, and `3=GBRG`.

Press `q`, Escape, close the window, or press Ctrl-C to stop an unlimited run.

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
