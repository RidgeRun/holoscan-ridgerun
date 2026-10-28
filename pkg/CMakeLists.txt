# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# add_holohub_package() handles the full cascade for both standalone
# packaging (`holoscan package holoscan-ridgerun --pkg-generator DEB`, which
# sets -DPKG_holoscan_ridgerun=ON via the HoloHub CLI) and in-tree
# HoloHub-monorepo packaging once this module is fetched as an external
# dependency.
add_holohub_package(
    holoscan-ridgerun
    OPERATORS
        argus_camera
        bgra_to_planar_tensor
        cuda_compositor
        drm_bgra_display
        drm_display
        fakesink_op
        flow_trace
        frame_skipper
        interprocess_src
        perf_op
        raw_bayer_to_bgra
        raw_bayer_to_yolo_tensor
        tensorrt_inference
        test_pattern
        v4l2_source
        v4l2_userptr_source
        x11_display
        yuyv_to_bgra
    APPLICATIONS
        argus_camera_drm
        argus_camera_x11
        interprocess
        ridgerun_pipeline
        v4l2_display
        v4l2_display_x11
        v4l2_imx274
        v4l2_imx274_drm
        v4l2_userptr_drm
        v4l2_userptr_perf
        v4l2_userptr_x11
        yolo11_userptr_inference
)
