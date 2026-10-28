# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Development container for holoscan-ridgerun.
#
# Extends the Holoscan SDK (HSDK) image with the build tools needed to configure,
# compile, and test the module from a live source mount. No source is copied
# into the image; the project tree is bind-mounted by the holoscan-cli at
# /workspace/holoscan_ridgerun when the container is launched.
#
# Build & run via the holoscan-cli wrapper (recommended):
#   ./holoscan_ridgerun run-container
#
# Manual build (without the CLI):
#   docker build --build-arg BASE_IMAGE=holoscan-sdk-build-$(uname -m):latest \
#       -t holoscan-holoscan-ridgerun .

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive

# Install the exact CLI pin with the create extra required by the wrapper.
# The wrapper checks that cookiecutter, jsonschema, and referencing import.
COPY requirements-cli.txt /tmp/requirements-cli.txt
RUN python3 -m pip install --no-cache-dir \
        --extra-index-url https://pypi.nvidia.com \
        -r /tmp/requirements-cli.txt \
    && holoscan version

# TODO: add module-specific runtime / build dependencies below.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        cmake \
        curl \
        libgtest-dev \
        clang-format \
        ninja-build \
        ccache \
    && rm -rf /var/lib/apt/lists/*
