# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(argus_camera_x11 argus_camera_x11.cpp)
target_compile_features(argus_camera_x11 PRIVATE cxx_std_20)
target_link_libraries(argus_camera_x11
    PRIVATE
        holoscan::argus_camera
        holoscan::x11_display
        holoscan::core
        Threads::Threads
)
