# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)

project(holoscan_ridgerun
    VERSION 0.1.0
    DESCRIPTION "RidgeRun Holoscan Module"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
find_package(holoscan 5.0.0 REQUIRED CONFIG COMPONENTS core sensor_io)

option(BUILD_ALL "Build all operators, applications, and packages" OFF)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HoloHubConfigHelpers)

# Packages are declared before their operator/application dependencies so the
# package helper can enable those subprojects when requested.
add_subdirectory(pkg)
add_subdirectory(applications)
add_subdirectory(operators)
