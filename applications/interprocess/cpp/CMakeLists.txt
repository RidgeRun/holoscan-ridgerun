# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(interprocess_sender interprocess_sender.cpp)
target_compile_features(interprocess_sender PRIVATE cxx_std_20)
target_link_libraries(interprocess_sender
    PRIVATE
        holoscan::core
        holoscan::raw_bayer_to_bgra
        holoscan::v4l2_userptr_source
        Threads::Threads
)

add_executable(interprocess_receiver interprocess_receiver.cpp)
target_compile_features(interprocess_receiver PRIVATE cxx_std_20)
target_link_libraries(interprocess_receiver
    PRIVATE
        holoscan::core
        holoscan::interprocess_src
        holoscan::x11_display
        Threads::Threads
)
