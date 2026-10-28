// SPDX-FileCopyrightText: Copyright (c) 2026 RidgeRun, LLC. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Minimal SDK 5 graph-validation application. Add the pipeline topology and
// run session here when the module is ready for execution.

#include <iostream>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <fakesink_op/fakesink_op.hpp>

namespace ops = holoscan::ops;

int main() {
  holoscan::Graph graph{"ridgerun_pipeline"};
  (void)graph.op<ops::FakesinkOp>("fakesink_op");
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  return 0;
}
