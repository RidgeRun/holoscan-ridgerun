# syntax=docker/dockerfile:1

# SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG BASE_IMAGE

FROM ${BASE_IMAGE} AS interprocess-base

COPY --chmod=755 holoscan_ridgerun /tmp/scripts/
RUN /tmp/scripts/holoscan_ridgerun env-info

FROM interprocess-base AS interprocess-sender

FROM interprocess-base AS interprocess-receiver

# X11DisplayOp is used only by the receiver process. Keep the sender image
# free of display-only runtime dependencies.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libx11-6 \
    && rm -rf /var/lib/apt/lists/*

FROM interprocess-base AS interprocess-validate
