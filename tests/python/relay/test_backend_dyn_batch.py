# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np
import pytest

import tvm
from tvm import relay
import tvm.testing


def test_with_params():
    def check_with(*, size=None, static_batch=None):
        # r1 = (x + transpose(y)) * z
        # r2 = transpose(res1 - 1)
        x = relay.var("x", shape=(static_batch, size))
        y = relay.var("y", shape=(size, static_batch))
        z = relay.var("z", shape=(1, size))
        one = relay.const(float(1))

        r1 = relay.multiply(relay.add(x, relay.transpose(y)), z)
        r2 = relay.transpose(relay.subtract(r1, one))
        func = relay.Function([x, y, z], relay.expr.Tuple([r1, r2]))

        lib = relay.build(tvm.IRModule.from_expr(func), "llvm")
        dyn_batch_conf = [(0, 0, False), (1, 1, False), (0, 0, True), (1, 1, True)]
        dyn_batch_wrp = tvm.get_global_func("tvm.graph_executor_dyn_batch.create")
        orig_g_mod = lib["default"](tvm.cpu(0))

        def run_with_batch(batch, mod):
            x_data = np.random.rand(batch, size).astype("float32")
            y_data = np.random.rand(size, batch).astype("float32")
            z_data = np.random.rand(1, size).astype("float32")

            x_data_tvm = tvm.nd.array(x_data)
            y_data_tvm = tvm.nd.array(y_data)
            z_data_tvm = tvm.nd.array(z_data)

            wrap_g_mod = dyn_batch_wrp(mod, dyn_batch_conf)
            set_input = wrap_g_mod["set_input"]
            get_output = wrap_g_mod["get_output"]
            run = wrap_g_mod["run"]

            set_input("x", x_data_tvm)
            set_input("y", y_data_tvm)
            set_input("z", z_data_tvm)
            run()
            res1 = get_output(0).numpy()
            res2 = get_output(1).numpy()

            ref_res1 = (x_data + np.transpose(y_data)) * z_data
            ref_res2 = np.transpose(ref_res1 - 1)
            tvm.testing.assert_allclose(res1, ref_res1, atol=1e-5, rtol=1e-5)
            tvm.testing.assert_allclose(res2, ref_res2, atol=1e-5, rtol=1e-5)

        run_with_batch(static_batch, orig_g_mod)
        run_with_batch(2*static_batch, orig_g_mod)
        # run_with_batch(int(2.5 * static_batch), orig_g_mod)  # TODO: have to support
        # run_with_batch(int(0.75 * static_batch), orig_g_mod)

    check_with(size=5, static_batch=10)
    check_with(size=32, static_batch=10)


if __name__ == "__main__":
    pytest.main([__file__])
