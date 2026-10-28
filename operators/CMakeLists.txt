# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared implementation support used by the native Holoscan SDK (HSDK) 5 operators.
add_subdirectory(operator_utils)
add_holohub_operator(flow_trace)
if(NOT TARGET flow_trace)
  add_subdirectory(flow_trace)
endif()

add_holohub_operator(argus_camera)
add_holohub_operator(bgra_to_planar_tensor)
add_holohub_operator(cuda_compositor)
add_holohub_operator(drm_display)
add_holohub_operator(drm_bgra_display DEPENDS OPERATORS drm_display)
add_holohub_operator(frame_skipper)
add_holohub_operator(raw_bayer_to_bgra)
add_holohub_operator(raw_bayer_to_yolo_tensor)
add_holohub_operator(tensorrt_inference)
add_holohub_operator(v4l2_source)
add_holohub_operator(v4l2_userptr_source)
add_holohub_operator(x11_display)
add_holohub_operator(yuyv_to_bgra)
add_holohub_operator(interprocess_src)
add_holohub_operator(test_pattern)
add_holohub_operator(fakesink_op)
add_holohub_operator(perf_op)
