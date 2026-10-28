// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "interprocess_src/interprocess_reader.hpp"

#include <stdexcept>
#include <utility>

namespace holoscan::ops {

InterprocessReader::InterprocessReader(
    holoscan::ChannelRuntime& runtime, holoscan::ChannelEndpointCatalog catalog,
    std::string endpoint_name)
    : runtime_(&runtime), catalog_(std::move(catalog)), endpoint_name_(std::move(endpoint_name)) {
  if (endpoint_name_.empty()) {
    throw std::invalid_argument("InterprocessReader endpoint name must not be empty");
  }
}

holoscan::expected<void, holoscan::Error> InterprocessReader::start() {
  if (runtime_ == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotReady, "InterprocessReader has no runtime"});
  }
  if (subscriber_) {
    return {};
  }
  auto subscriber = runtime_->create_subscriber<holoscan::Tensor>(catalog_, endpoint_name_);
  if (!subscriber) {
    return holoscan::make_unexpected(std::move(subscriber).error());
  }
  auto refreshed = subscriber->refresh_connections();
  if (!refreshed) {
    return holoscan::make_unexpected(std::move(refreshed).error());
  }
  subscriber_.emplace(std::move(*subscriber));
  return {};
}

void InterprocessReader::stop() noexcept { subscriber_.reset(); }

holoscan::expected<std::optional<holoscan::Sample<holoscan::Tensor>>, holoscan::Error>
InterprocessReader::receive() {
  if (!subscriber_) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kNotReady, "InterprocessReader is not started"});
  }
  auto sample = subscriber_->receive_sample();
  if (!sample) {
    if (sample.error().code == holoscan::ErrorCode::kNotReady) {
      return std::optional<holoscan::Sample<holoscan::Tensor>>{};
    }
    return holoscan::make_unexpected(std::move(sample).error());
  }
  return std::optional<holoscan::Sample<holoscan::Tensor>>{std::move(*sample)};
}

}  // namespace holoscan::ops
