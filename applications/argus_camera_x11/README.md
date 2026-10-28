# Argus camera to X11

This Holoscan SDK (HSDK) 5 C++ application captures an Argus-compatible MIPI CSI-2
camera connected directly to a Jetson board and displays it in an X11 window.
It is originally tested on a Jetson Orin AGX with JetPack 7.2.1 / Jetson Linux 39.2.1.

The graph is deliberately short:

```text
libArgus + ISP -> NV12 EGLStream/CUDA arrays -> CUDA BGRA tensor -> X11DisplayOp
```

## Memory path

The camera operator reads the Y and UV CUDA arrays provided by the Argus
EGLStream directly in its color-conversion kernel. This avoids copying those
planes into a temporary linear NV12 buffer before conversion.

There is one explicit memory transfer per displayed frame:

- Argus/ISP to EGLStream: no CPU copy.
- EGLStream CUDA arrays to BGRA: one CUDA kernel write, with no staging copy.
- BGRA to X11 `XPutImage`: one device-to-host copy, performed by the existing
  `X11DisplayOp` on the runtime CUDA stream.

Plain Xlib `XPutImage` cannot consume CUDA device memory, so that final transfer
is unavoidable for this X11 sink. The queue depth is one with drop-oldest
semantics to avoid latency growth.

## Prerequisites

Install the JetPack development components and X11 headers. NVIDIA documents
`nvidia-jetpack-dev` as the development metapackage; the complete
`nvidia-jetpack` metapackage also works.

```bash
sudo apt update
sudo apt install nvidia-jetpack-dev libx11-dev
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
./holoscan_ridgerun build argus_camera_x11 live \
  --local --language cpp
./holoscan_ridgerun run argus_camera_x11 validate \
  --local --language cpp --no-local-build \
  --run-args='--width 1920 --height 1080'
```

Validation compiles the graph without opening the camera or X11 window.

## Run

Use the exact width, height, and supported frame rate shown by
`nvargus_nvraw --lps` for the selected mode. For example:

```bash
export DISPLAY=:0
./holoscan_ridgerun run argus_camera_x11 live \
  --local --language cpp --no-local-build \
  --run-args='--camera 0 --sensor-mode 0 --width 1920 --height 1080 --fps 60 --frame-limit 600'
```

Omit `--frame-limit` for an unlimited run.

The operator intentionally rejects a width/height or frame rate that does not
match the selected Argus sensor mode. Press `q`, Escape, or Ctrl-C to stop.

## References

- [HSDK 5.0 source](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v5.0.0)
- [Jetson Linux 39.2 Argus EGL output stream API](https://docs.nvidia.com/jetson/archives/r39.2/ApiReference/group__ArgusEGLOutputStream.html)
- [JetPack component installation](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html)
