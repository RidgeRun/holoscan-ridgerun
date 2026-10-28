# Argus camera to DRM/KMS

This Holoscan SDK (HSDK) 5 C++ application captures an Argus-compatible MIPI CSI-2
camera connected directly to a Jetson board and displays it through an exclusive DRM/KMS output.
It is originally tested on a Jetson Orin AGX with JetPack 7.2.1 / Jetson Linux 39.2.1.

The graph is deliberately short:

```text
libArgus + ISP -> NV12 EGLStream/CUDA arrays -> CUDA BGRA tensor -> DrmBgraDisplayOp
```

## Memory path

- Argus/ISP to EGLStream: no CPU copy.
- EGLStream CUDA arrays to BGRA: one CUDA kernel write, without NV12 staging.
- BGRA to DRM: the Orin-optimized DRM operator writes into a double-buffered scanout framebuffer. It prefers a CUDA-registered DRM mapping and retains a direct mmap fallback.
- Queue depth is one with drop-oldest semantics to prevent latency accumulation.

## Prerequisites

Install the JetPack development components and DRM headers. NVIDIA documents
`nvidia-jetpack-dev` as the development metapackage, the complete
`nvidia-jetpack` metapackage also works.

```bash
sudo apt update
sudo apt install nvidia-jetpack-dev libdrm-dev
```

The following files must then exist:

```text
/usr/src/jetson_multimedia_api/argus/include/Argus/Argus.h
/usr/lib/aarch64-linux-gnu/nvidia/libnvargus_socketclient.so
```

The camera must already be described by the device tree and recognized by
Argus. List camera IDs, sensor modes, dimensions, and frame rates with:

```bash
nvargus_nvraw --lps
```

Keep `nvargus-daemon` running; the application links the supported multi-process
Argus socket client.

## Build and validate

From the repository root:

```bash
./holoscan_ridgerun build argus_camera_drm live \
  --local --language cpp
./holoscan_ridgerun run argus_camera_drm validate \
  --local --language cpp --no-local-build \
  --run-args='--width 1920 --height 1080'
```

Validation compiles the graph without opening the camera or DRM device.

## Run

Use the exact sensor mode reported by `nvargus_nvraw --lps`:

```bash
sudo systemctl stop display-manager
./holoscan_ridgerun run argus_camera_drm live \
  --local --language cpp --no-local-build \
  --run-args='--camera 0 --sensor-mode 0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

Omit `--frame-limit` for an unlimited run. Press Ctrl-C to stop manually. The
previous CRTC is restored during normal shutdown. Restore the desktop afterward
with `sudo systemctl start display-manager`. The host `nvargus-daemon` must
remain running.

## References

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
- [Jetson Linux 39.2 Argus EGL output stream API](https://docs.nvidia.com/jetson/archives/r39.2/ApiReference/group__ArgusEGLOutputStream.html)
- [JetPack component installation](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html)
