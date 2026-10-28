# V4L2 RG10 to DRM/KMS

This Holoscan SDK (HSDK) 5 C++ application captures RG10 Bayer frames from an
IMX274-compatible V4L2 device, converts them to BGRA on CUDA, and presents them
through an exclusive DRM/KMS output. It supports MMAP and CUDA-mapped USERPTR paths.

## Pipeline

```text
V4L2SourceOp (MMAP) or V4L2UserPtrSourceOp (USERPTR)
  -> RawBayerToBgraOp
  -> DrmBgraDisplayOp
```

Both capture paths emit a CUDA-visible `uint16 [height,width,1]` RG10 tensor.
`RawBayerToBgraOp` performs Bayer demosaicing, black-level correction, digital
gain, gamma encoding, and BGRA conversion on CUDA. `DrmBgraDisplayOp` presents
the BGRA tensor through DRM/KMS.

Connections use a queue depth of one with drop-oldest behavior to avoid latency
accumulation. The application can stop after a duration; it does not provide a
frame-limit option.

## Prerequisites

The camera must expose RG10 Bayer at the requested dimensions and frame rate.
Inspect its modes with:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Install the V4L2 and DRM development utilities when building natively:

```bash
sudo apt update
sudo apt install v4l-utils libdrm-dev
```

The process must acquire DRM master. Stop the desktop display manager from SSH
or a TTY before a live run:

```bash
sudo systemctl stop display-manager
```

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_display live --local --language cpp
./holoscan_ridgerun run v4l2_display validate \
  --local --language cpp --no-local-build
```

Validation compiles the graph without opening the camera or DRM device.

## Run

The dimensions passed to the camera and DRM mode must match:

```bash
./holoscan_ridgerun run v4l2_display live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --capture-memory mmap --width 1920 --height 1080 --fps 60 --optical-black 50 --raw-shift 6 --display-gain 12.0 --gamma 2.2 --bayer-grid 1 --drm-device /dev/dri/card2 --mode 1920x1080 --duration 30'
```

Use `--capture-memory userptr` to select CUDA-mapped USERPTR capture. Adjust
`--optical-black`, `--display-gain`, and `--gamma` for the sensor and lighting.
The Bayer-grid values are `0=BGGR`, `1=RGGB`, `2=GRBG`, and `3=GBRG`.
Use `--connector ID` when automatic connector selection is not appropriate.

Restore the desktop after the application exits:

```bash
sudo systemctl start display-manager
```

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
