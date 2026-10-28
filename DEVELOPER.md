# Developer Guide — holoscan-ridgerun

This repository is a collection of operators and applications based on
Holoscan SDK (HSDK) version 5. Python is used only by the Holoscan CLI launcher;
the applications, operators, and CUDA kernels are implemented in C++.

## Module layout

```text
holoscan-ridgerun/
├── holoscan_ridgerun                         # Holoscan CLI launcher
├── metadata.json                             # Module descriptor
├── applications/                             # Registered application graphs
│   ├── argus_camera_drm/
│   ├── argus_camera_x11/
│   ├── interprocess/
│   ├── ridgerun_pipeline/
│   ├── v4l2_display/
│   ├── v4l2_display_x11/
│   ├── v4l2_imx274/
│   ├── v4l2_imx274_drm/
│   ├── v4l2_userptr_drm/
│   ├── v4l2_userptr_x11/
│   └── yolo11_userptr_inference/
├── operators/                                # Reusable C++ operators
├── cmake/                                    # HoloHub CMake helpers
└── pkg/                                      # Debian package configuration
```

The aggregate registration files are the source of truth for build
relationships:

- [`applications/CMakeLists.txt`](applications/CMakeLists.txt) registers each
  application and its operator dependencies.
- [`operators/CMakeLists.txt`](operators/CMakeLists.txt) registers each
  discoverable operator and the shared V4L2 support target.
- [`pkg/CMakeLists.txt`](pkg/CMakeLists.txt) includes the complete module in a
  Debian package.

## Holoscan CLI workflow

The launcher selects the SDK installation, prepares the CLI environment, and
forwards commands to `holoscan-cli`:

```bash
./holoscan_ridgerun build v4l2_display_x11
./holoscan_ridgerun run v4l2_display_x11
./holoscan_ridgerun build yolo11_userptr_inference
```

Set `HOLOSCAN_SDK_INSTALL_DIR` or `HOLOSCAN_SDK_ROOT` when the SDK is not in a
sibling checkout. The default CLI search path is the module root so the
recursive application and operator metadata is discoverable.

## CMake workflow

With an installed Holoscan SDK (HSDK) and a discoverable CMake package:

```bash
cmake -S . -B build -DBUILD_ALL=ON
cmake --build build -j"$(nproc)"
```

Individual projects can be selected with the corresponding `APP_<name>` or
`OP_<name>` CMake option when `BUILD_ALL` is disabled. Dependencies declared by
`DEPENDS OPERATORS` are enabled by the HoloHub helpers.

## Validation before a change is submitted

Run the format, spelling, Markdown, and pre-commit checks configured by the
repository:

```bash
pre-commit install
pre-commit run --all-files
```

Also verify that application and operator metadata remain valid JSON and that
the application graph still selects the intended source, conversion, display,
or inference path.
