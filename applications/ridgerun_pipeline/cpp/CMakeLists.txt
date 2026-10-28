# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(ridgerun_pipeline ridgerun_pipeline.cpp)

target_link_libraries(ridgerun_pipeline PRIVATE
    holoscan::fakesink_op
    holoscan::core)

install(TARGETS ridgerun_pipeline RUNTIME DESTINATION bin)
