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

#include "sut.hpp"

#include <tvm/runtime/module.h>



class TVMSUT : public SUT {
  public:
    TVMSUT(tvm::runtime::Module lib, long macs) : lib_(lib), macs(macs) {}
    
    virtual void init(int idx) {
        gmod_ = mod_factory.GetFunction("default")(dev);
        run_ = gmod.GetFunction("run");
        auto set_input = gmod.GetFunction("set_input");
        

    }

    void action() override {
        run_();
    }

    void finalize() override {
        // NOTHING
    }

  private:
    tvm::runtime::Module lib_;
    tvm::runtime::Module gmod_;
    tvm::runtime::PackedFunc run_;
}

std::shared_ptr<SUT> constructSUT_TVM() {
  DLDevice dev{kDLCPU, 0};
  auto lib = tvm::runtime::Module::LoadFromFile(path);

  return std::make_shared<TVMSUT>(lib, 100500);
}