# IMX274 V4L2 capture to DRM/KMS

This Holoscan SDK (HSDK) 5 C++ application captures an IMX274 RG10 Bayer stream
through V4L2 MMAP, processes it on CUDA, and displays the resulting BGRA image
through an exclusive DRM/KMS output.

## Pipeline

```text
V4L2SourceOp (RG10 MMAP) -> RawBayerToBgraOp -> DrmBgraDisplayOp
```

`V4L2SourceOp` copies each dequeued MMAP frame into a plan-owned CUDA tensor.
`RawBayerToBgraOp` performs Bayer reconstruction and display-oriented image
processing on CUDA. `DrmBgraDisplayOp` presents the BGRA output through a
double-buffered DRM scanout path.

Connections use a queue depth of one with drop-oldest behavior.

## Prerequisites

Confirm that the camera exposes RG10 at the requested mode:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Install `v4l-utils` and `libdrm-dev` for a native build. The application must
acquire DRM master, so stop the desktop from SSH or a TTY before running it:

```bash
sudo systemctl stop display-manager
```

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_imx274_drm live --local --language cpp
./holoscan_ridgerun run v4l2_imx274_drm validate \
  --local --language cpp --no-local-build
```

## Run

The DRM mode must match the frame dimensions:

```bash
./holoscan_ridgerun run v4l2_imx274_drm live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

Use `--connector ID` to select a specific connector. Restore the desktop after
the application exits:

```bash
sudo systemctl start display-manager
```

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
