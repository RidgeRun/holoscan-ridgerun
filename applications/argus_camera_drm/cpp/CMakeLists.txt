# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(argus_camera_drm argus_camera_drm.cpp)
target_compile_features(argus_camera_drm PRIVATE cxx_std_20)
target_link_libraries(argus_camera_drm
    PRIVATE
        holoscan::argus_camera
        holoscan::drm_bgra_display
        holoscan::drm_display
        holoscan::core
        Threads::Threads
)
