# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Each application is gated on its own APP_<name> option, defaulting to ${BUILD_ALL}.
# Add one add_holohub_application() call per application; declare operator
# dependencies via DEPENDS OPERATORS so they auto-enable when the app is built.
add_holohub_application(argus_camera_x11
    DEPENDS OPERATORS argus_camera x11_display
)

add_holohub_application(argus_camera_drm
    DEPENDS OPERATORS argus_camera drm_bgra_display drm_display
)

add_holohub_application(yolo11_userptr_inference
    DEPENDS OPERATORS fakesink_op tensorrt_inference v4l2_userptr_source
)

add_holohub_application(v4l2_imx274
    DEPENDS OPERATORS raw_bayer_to_bgra v4l2_source x11_display
)

add_holohub_application(v4l2_userptr_x11
    DEPENDS OPERATORS raw_bayer_to_bgra v4l2_userptr_source x11_display
)

add_holohub_application(v4l2_userptr_perf
    DEPENDS OPERATORS perf_op raw_bayer_to_bgra v4l2_userptr_source x11_display
)

add_holohub_application(v4l2_userptr_drm
    DEPENDS OPERATORS
        drm_bgra_display drm_display raw_bayer_to_bgra v4l2_userptr_source
)

add_holohub_application(v4l2_imx274_drm
    DEPENDS OPERATORS drm_bgra_display drm_display raw_bayer_to_bgra v4l2_source
)

add_holohub_application(interprocess
    DEPENDS OPERATORS interprocess_src raw_bayer_to_bgra v4l2_userptr_source x11_display
)

add_holohub_application(v4l2_display
    DEPENDS OPERATORS
        drm_bgra_display drm_display raw_bayer_to_bgra v4l2_source v4l2_userptr_source
)

add_holohub_application(v4l2_display_x11
    DEPENDS OPERATORS raw_bayer_to_bgra v4l2_source v4l2_userptr_source x11_display
)

add_holohub_application(ridgerun_pipeline
    DEPENDS OPERATORS fakesink_op
)
