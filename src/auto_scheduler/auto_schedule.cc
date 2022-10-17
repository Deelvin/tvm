/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file auto_scheduler/auto_schedule.cc
 * \brief The user interface and tuning options of the TVM auto-scheduler.
 */

#include <tvm/auto_scheduler/auto_schedule.h>
#include <tvm/runtime/registry.h>

#include <tvm/driver/driver_api.h>
#include <tvm/te/operation.h>
#include <tvm/te/tensor.h>

#include "utils.h"

namespace tvm {
namespace auto_scheduler {

TVM_REGISTER_NODE_TYPE(TuningOptionsNode);

TuningOptions::TuningOptions(int num_measure_trials, int early_stopping, int num_measures_per_round,
                             int verbose, ProgramBuilder builder, ProgramRunner runner,
                             Optional<Array<MeasureCallback>> measure_callbacks) {
  auto node = make_object<TuningOptionsNode>();
  node->num_measure_trials = num_measure_trials;
  node->early_stopping = early_stopping;
  node->num_measures_per_round = num_measures_per_round;
  node->verbose = verbose;
  node->builder = std::move(builder);
  node->runner = std::move(runner);
  node->measure_callbacks = std::move(measure_callbacks);
  data_ = std::move(node);
}

std::pair<te::Schedule, Array<te::Tensor>> AutoSchedule(SearchPolicy search_policy,
                                                        TuningOptions tuning_options) {
  // Create a ProgramMeasurer to handle the schedule build and performance measure
  ProgramMeasurer measurer =
      ProgramMeasurer(tuning_options->builder, tuning_options->runner,
                      tuning_options->measure_callbacks, tuning_options->verbose);
  // Search for the best schedule
  std::cout << "ICE Search" << std::endl;
  State state =
      search_policy->Search(tuning_options->num_measure_trials, tuning_options->early_stopping,
                            tuning_options->num_measures_per_round, measurer);
  if (state.defined()) {
    // std::cout << "ICE Search ApplySteps state.defined()" << std::endl;


    std::pair<te::Schedule, Array<te::Tensor>> r = search_policy->search_task->compute_dag.ApplySteps(state->transform_steps);
    // auto target = Target("llvm");
    // te::Schedule sch = r.first;
    // const Array<te::Tensor>& args = r.second;
    // const std::string& name = "func";
    // std::unordered_map<te::Tensor, Buffer> binds;

    // tvm::GlobalVarSupply global_var_supply = tvm::GlobalVarSupply(tvm::NameSupply(""));
    // bool simple_mode = true;

    // auto lowered = tvm::LowerSchedule(sch, args, name, binds, global_var_supply, simple_mode);
    // // auto module = tvm::build(lowered, target, Target());
    
    // std::cout << "ICE schedule_lowered\n " << lowered << std::flush << std::endl;
    
    return r;
  } else {
    StdCout(tuning_options->verbose)
        << "No valid state found in this search round. Check if it has traversed all of the "
        << "search space." << std::endl;
    // std::cout << "ICE Search ApplySteps state.defined() == false" << std::endl;


    std::pair<te::Schedule, Array<te::Tensor>> r = {te::Schedule(search_policy->search_task->compute_dag->ops),
            search_policy->search_task->compute_dag->tensors};
    // auto target = Target("llvm");
    // te::Schedule sch = r.first;
    // const Array<te::Tensor>& args = r.second;
    // const std::string& name = "func";
    // std::unordered_map<te::Tensor, Buffer> binds;

    // tvm::GlobalVarSupply global_var_supply = tvm::GlobalVarSupply(tvm::NameSupply(""));
    // bool simple_mode = true;

    // auto lowered = tvm::LowerSchedule(sch, args, name, binds, global_var_supply, simple_mode);
    // // auto module = tvm::build(lowered, target, Target());
    
    // std::cout << "ICE schedule_lowered\n " << lowered << std::flush << std::endl;

    // Return the default schedule
    return r;
  }
}

TVM_REGISTER_GLOBAL("auto_scheduler.TuningOptions")
    .set_body_typed([](int num_measure_trials, int early_stopping, int num_measures_per_round,
                       int verbose, ProgramBuilder builder, ProgramRunner runner,
                       Optional<Array<MeasureCallback>> measure_callbacks) {
      return TuningOptions(num_measure_trials, early_stopping, num_measures_per_round, verbose,
                           builder, runner, measure_callbacks);
    });

TVM_REGISTER_GLOBAL("auto_scheduler.AutoSchedule")
    .set_body_typed([](SearchPolicy search_policy, TuningOptions tuning_options) {
      auto [sch, return_tensors] = AutoSchedule(search_policy, tuning_options);
      return Array<ObjectRef>{sch, return_tensors};
    });
}  // namespace auto_scheduler
}  // namespace tvm
