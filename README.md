# holoscan-ridgerun

`holoscan-ridgerun` is a collection of operators and applications for camera capture,
format conversion, display, inter-process video, and TensorRT inference. It
uses the Holoscan SDK (HSDK) 5 C++ graph APIs and follows the same application/operator
layout as the surrounding HoloHub modules.

## Applications

The module contains the camera and inference applications below. Each
application is independently selectable through the Holoscan CLI or its
CMake option.

| Application | Description | Main path |
| --- | --- | --- |
| `argus_camera_drm` | Captures an Argus-compatible CSI camera and displays it through DRM/KMS. | Argus → CUDA BGRA → `drm_bgra_display` |
| `argus_camera_x11` | Captures an Argus-compatible CSI camera and displays it in an X11 window. | Argus → CUDA BGRA → `x11_display` |
| `v4l2_display` | Captures RG10 Bayer frames with selectable MMAP or USERPTR memory and displays them through DRM/KMS. | V4L2 → `raw_bayer_to_bgra` → `drm_bgra_display` |
| `v4l2_display_x11` | Captures RG10 Bayer frames with selectable MMAP or USERPTR memory and displays them through X11. | V4L2 → `raw_bayer_to_bgra` → `x11_display` |
| `v4l2_imx274` | Captures the IMX274 raw Bayer stream and displays it through X11. | V4L2 → `raw_bayer_to_bgra` → `x11_display` |
| `v4l2_imx274_drm` | Captures the IMX274 raw Bayer stream and displays it through DRM/KMS. | V4L2 → `raw_bayer_to_bgra` → `drm_bgra_display` |
| `v4l2_userptr_x11` | Captures with the V4L2 USERPTR path and displays through X11. | USERPTR → `raw_bayer_to_bgra` → `x11_display` |
| `v4l2_userptr_drm` | Captures with the V4L2 USERPTR path and displays through DRM/KMS. | USERPTR → `raw_bayer_to_bgra` → `drm_bgra_display` |
| `v4l2_userptr_perf` | Captures through V4L2 USERPTR and reports live performance statistics before X11 display. | USERPTR → `raw_bayer_to_bgra` → `perf_op` → `x11_display` |
| `yolo11_userptr_inference` | Captures IMX274 RG10 frames, creates the YOLO11 tensor on CUDA, and runs TensorRT inference headlessly. | USERPTR → fused Bayer tensor → `tensorrt_inference` → `fakesink_op` |
| `interprocess` | Sends and receives CUDA BGRA tensors through Holoscan Interprocess endpoints. | USERPTR → `raw_bayer_to_bgra` → HoloMQ → `interprocess_src` → `x11_display` |
| `ridgerun_pipeline` | Minimal graph-validation application used to exercise the local sink operator. | `fakesink_op` |


Application-specific prerequisites, graph details, model settings, and run
arguments are documented in the README files inside the applications that
provide them. The aggregate build registrations are in
[`applications/CMakeLists.txt`](applications/CMakeLists.txt).

## Operators

The module provides the following operators and shared implementation support:

| Group | Operators |
| --- | --- |
| Camera sources | `argus_camera`, `v4l2_source`, `v4l2_userptr_source` |
| CUDA conversion | `bgra_to_planar_tensor`, `raw_bayer_to_bgra`, `raw_bayer_to_yolo_tensor`, `yuyv_to_bgra` |
| Inference | `tensorrt_inference` |
| Display | `drm_display`, `drm_bgra_display`, `x11_display` |
| Utility and test | `fakesink_op`, `flow_trace`, `frame_skipper`, `interprocess_src`, `perf_op`, `test_pattern`, `cuda_compositor` |


Application/operator relationships are declared explicitly with
`DEPENDS OPERATORS` in the two aggregate CMake files. Selecting an application
therefore enables the operators needed by its graph; the module does not rely
on implicit or duplicate copies of those implementations.

### Code origin and attribution

The attribution below is based on comparison with the upstream Holoscan 5
HoloHub source tree and this repository's development history.

| Origin | Operators and support code |
| --- | --- |
| NVIDIA-derived | `bgra_to_planar_tensor`, `cuda_compositor`, `flow_trace`, `frame_skipper`, `operator_utils`, `raw_bayer_to_bgra`, `tensorrt_inference`, `test_pattern`, `v4l2_source`, `x11_display`, `yuyv_to_bgra` |
| RidgeRun | `argus_camera`, `drm_bgra_display`, `drm_display`, `fakesink_op`, `frame_limit`, `interprocess_src`, `perf_op`, `raw_bayer_to_yolo_tensor`, `v4l2_userptr_source` |

The applications in this module were assembled and integrated by RidgeRun.
NVIDIA-derived implementation files retain NVIDIA attribution. RidgeRun-owned
files and metadata use RidgeRun attribution; files containing substantial code
from both origins may carry both notices.

## Build and run

Build HSDK first, or provide an existing SDK installation through
`HOLOSCAN_SDK_INSTALL_DIR`. The launcher can discover a sibling SDK checkout
with the usual `holoscan-sdk/install-<arch>` layout.

Install the pinned Holoscan CLI in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  -r requirements-cli.txt
```

Build and run an application with the module launcher:

```bash
./holoscan_ridgerun build argus_camera_x11
./holoscan_ridgerun run argus_camera_x11
```

Replace `argus_camera_x11` with any application in the tables above. For a
local CMake build with an installed SDK:

```bash
cmake -S . -B build -DBUILD_ALL=ON
cmake --build build -j"$(nproc)"
```

The module's package declaration includes all applications and registered
operators. Package it from the module root when a Debian artifact is needed:

```bash
./holoscan_ridgerun package holoscan-ridgerun --pkg-generator DEB
```

## Repository layout

```text
holoscan-ridgerun/
├── applications/          # Camera, display, Interprocess, pipeline, and YOLO graphs
├── operators/             # Reusable C++ operators and CUDA kernels
├── cmake/                 # HoloHub CMake helpers
├── pkg/                   # Debian package configuration
├── holoscan_ridgerun      # Holoscan CLI launcher
└── metadata.json          # Module descriptor
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
