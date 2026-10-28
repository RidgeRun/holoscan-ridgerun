# IMX274 V4L2 USERPTR capture to DRM/KMS

This Holoscan SDK (HSDK) 5 C++ application captures an IMX274 RG10 Bayer stream
into CUDA-mapped V4L2 USERPTR buffers, processes it on CUDA, and displays the
resulting BGRA image through an exclusive DRM/KMS output.

## Pipeline

```text
V4L2UserPtrSourceOp -> RawBayerToBgraOp -> DrmBgraDisplayOp
```

The capture slots are pinned host allocations queued to V4L2 as USERPTR and
mapped into the CUDA address space. The capture operator therefore avoids an
explicit host-to-device frame copy. USERPTR support depends on the V4L2 driver.

`RawBayerToBgraOp` generates the BGRA image on CUDA and
`DrmBgraDisplayOp` presents it through a double-buffered DRM scanout path.
Connections use a queue depth of one with drop-oldest behavior.

## Prerequisites

Verify that the camera supports RG10 and V4L2 USERPTR for the selected mode:

```bash
v4l2-ctl --device /dev/video0 --list-formats-ext
```

Install `v4l-utils` and `libdrm-dev` for a native build. Stop the desktop
display manager from SSH or a TTY so the application can acquire DRM master:

```bash
sudo systemctl stop display-manager
```

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build v4l2_userptr_drm live --local --language cpp
./holoscan_ridgerun run v4l2_userptr_drm validate \
  --local --language cpp --no-local-build
```

## Run

The DRM mode must match the captured frame dimensions:

```bash
./holoscan_ridgerun run v4l2_userptr_drm live \
  --local --language cpp --no-local-build \
  --run-args='--device /dev/video0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

Use `--connector ID` for an explicit DRM connector. Restore the desktop after
the application exits:

```bash
sudo systemctl start display-manager
```

## Reference

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
