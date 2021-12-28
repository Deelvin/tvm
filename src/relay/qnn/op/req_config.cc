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
 * \file src/relay/qnn/op/req_config.cc
 * \brief QNN requantize config.
 */

#include "./req_config.h"

#include <dmlc/thread_local.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/relay/transform.h>

#include <stack>

namespace tvm {
namespace relay {
namespace qnn {

/*! \brief Entry to hold the BuildConfig context stack. */
struct TVMReqConfigThreadLocalEntry {
  /*! \brief The default build config if the stack is empty */
  ReqConfig default_config;

  /*! \brief The current build config context */
  std::stack<ReqConfig> context_stack;

  TVMReqConfigThreadLocalEntry() : default_config(make_object<ReqConfigNode>(true)) {}
};

/*! \brief Thread local store to hold the BuildConfig context stack. */
typedef dmlc::ThreadLocalStore<TVMReqConfigThreadLocalEntry> TVMReqConfigThreadLocalStore;

void ReqConfig::EnterReqConfigScope(const ReqConfig& build_config) {
  TVMReqConfigThreadLocalEntry* entry = TVMReqConfigThreadLocalStore::Get();
  entry->context_stack.push(build_config);
}

void ReqConfig::ExitReqConfigScope() {
  TVMReqConfigThreadLocalEntry* entry = TVMReqConfigThreadLocalStore::Get();
  entry->context_stack.pop();
}

ReqConfig& ReqConfig::Current() {
  TVMReqConfigThreadLocalEntry* entry = TVMReqConfigThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }

  return entry->default_config;
}

TVM_REGISTER_NODE_TYPE(ReqConfigNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ReqConfigNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* op = static_cast<const ReqConfigNode*>(ref.get());
      p->stream << "reqconfig(";
      p->stream << "rounding==" << op->get_rounding() << ", ";
      p->stream << "calculation_flow_type==" << op->get_calculation_flow_type();
      p->stream << ")";
    });

TVM_REGISTER_GLOBAL("relay._requantize._GetCurrentReqConfig").set_body_typed([]() -> ReqConfig {
  return ReqConfig::Current();
});

TVM_REGISTER_GLOBAL("relay._requantize._EnterReqConfigScope")
    .set_body_typed(ReqConfig::EnterReqConfigScope);

TVM_REGISTER_GLOBAL("relay._requantize._ExitReqConfigScope")
    .set_body_typed(ReqConfig::ExitReqConfigScope);

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
