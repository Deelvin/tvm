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
"""Minimum graph executor that executes graph containing TVM PackedFunc."""
import numpy as np
import tvm._ffi
from .graph_executor import GraphModule


def create(orig_g_mod, config="default"):
    if not isinstance(orig_g_mod, GraphModule):
        raise ValueError(
            "Wrong argument type. orig_g_mod expected to be graph_executor.GraphModule."
        )

    # if isinstance(config, str) and config == "default":
    #     dyn_batch_conf = [(i, 0, False) for i in range(orig_g_mod.get_num_inputs())]
    #     dyn_batch_conf += [(i, 0, True) for i in range(orig_g_mod.get_num_outputs())]

    dyn_batch_factory = tvm.get_global_func("tvm.graph_executor_dyn_batch.create")

    return DynBatchSlicer(dyn_batch_factory(orig_g_mod.module, config))


class DynBatchSlicer(object):
    """Wrapper for graph executor to handle tensors with arbitrary batch size.
    """

    def __init__(self, module):
        self.module = module
        self._set_input = module["set_input"]
        self._run = module["run"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_input_index = module["get_input_index"]
        self._get_num_inputs = module["get_num_inputs"]
        self._load_params = module["load_params"]
        self._share_params = module["share_params"]

    def set_input(self, key=None, value=None, **params):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : int or str
           The input key

        value : the input value.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        if key is not None:
            if isinstance(value, np.ndarray):
                value = tvm.nd.array(value)

            self._set_input(key, value)

        if params:
            for k, val in params.items():
                if isinstance(val, np.ndarray):
                    val = tvm.nd.array(val)

                self._set_input(k, val)

    def run(self, **input_dict):
        """Run forward execution of the graph

        Parameters
        ----------
        input_dict: dict of str to NDArray
            List of input values to be feed to
        """
        if input_dict:
            self.set_input(**input_dict)
        self._run()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the graph

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, index, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(index).copyto(out)
            return out

        return self._get_input(index)

    def get_input_index(self, name):
        """Get inputs index via input name.

        Parameters
        ----------
        name : str
           The input key name

        Returns
        -------
        index: int
            The input index. -1 will be returned if the given input name is not found.
        """
        return self._get_input_index(name)

    def get_output(self, index, out=None):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index

        out : NDArray
            The output array container
        """
        if out:
            self._get_output(index, out)
            return out

        return self._get_output(index)

    def debug_get_output(self, node, out):
        """Run graph up to node and get the output to out

        Parameters
        ----------
        node : int / str
            The node index or name

        out : NDArray
            The output array container
        """
        raise NotImplementedError("Please use debugger.debug_executor as graph_executor instead.")

    def load_params(self, params_bytes):
        """Load parameters from serialized byte array of parameter dict.

        Parameters
        ----------
        params_bytes : bytearray
            The serialized parameter dict.
        """
        self._load_params(bytearray(params_bytes))

    def share_params(self, other, params_bytes):
        """Share parameters from pre-existing GraphExecutor instance.

        Parameters
        ----------
        other: GraphExecutor
            The parent GraphExecutor from which this instance should share
            it's parameters.
        params_bytes : bytearray
            The serialized parameter dict (used only for the parameter names).
        """
        self._share_params(other.module, bytearray(params_bytes))

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]

    def benchmark(
        self,
        device,
        func_name="run",
        repeat=5,
        number=5,
        min_repeat_ms=None,
        end_to_end=False,
        **kwargs,
    ):
        """Calculate runtime of a function by repeatedly calling it.

        Use this function to get an accurate measurement of the runtime of a function. The function
        is run multiple times in order to account for variability in measurements, processor speed
        or other external factors.  Mean, median, standard deviation, min and max runtime are all
        reported.  On GPUs, CUDA and ROCm specifically, special on-device timers are used so that
        synchonization and data transfer operations are not counted towards the runtime. This allows
        for fair comparison of runtimes across different functions and models. The `end_to_end` flag
        switches this behavior to include data transfer operations in the runtime.

        The benchmarking loop looks approximately like so:

        .. code-block:: python

            for r in range(repeat):
                time_start = now()
                for n in range(number):
                    func_name()
                time_end = now()
                total_times.append((time_end - time_start)/number)


        Parameters
        ----------
        func_name : str
            The function to benchmark. This is ignored if `end_to_end` is true.

        repeat : int
            Number of times to run the outer loop of the timing code (see above). The output will
            contain `repeat` number of datapoints.

        number : int
            Number of times to run the inner loop of the timing code. This inner loop is run in
            between the timer starting and stopping. In order to amortize any timing overhead,
            `number` should be increased when the runtime of the function is small (less than a 1/10
            of a millisecond).

        min_repeat_ms : Optional[float]
            If set, the inner loop will be run until it takes longer than `min_repeat_ms`
            milliseconds. This can be used to ensure that the function is run enough to get an
            accurate measurement.

        end_to_end : bool
            If set, include time to transfer input tensors to the device and time to transfer
            returned tensors in the total runtime. This will give accurate timings for end to end
            workloads.

        kwargs : Dict[str, Object]
            Named arguments to the function. These are cached before running timing code, so that
            data transfer costs are not counted in the runtime.

        Returns
        -------
        timing_results : BenchmarkResult
            Runtimes of the function. Use `.mean` to access the mean runtime, use `.results` to
            access the individual runtimes (in seconds).
        """
        min_repeat_ms = 0 if min_repeat_ms is None else min_repeat_ms
        if end_to_end:
            # Have to unpack kwargs into a single list
            args = []
            for k, v in kwargs.items():
                args.append(k)
                args.append(v)
            return self.module.time_evaluator(
                "run_from_inputs",
                device,
                repeat=repeat,
                number=number,
                min_repeat_ms=min_repeat_ms,
            )(device.device_type % rpc_base.RPC_SESS_MASK, device.device_id, *args)
        if kwargs:
            self.set_input(**kwargs)
        return self.module.time_evaluator(
            func_name, device, repeat=repeat, number=number, min_repeat_ms=min_repeat_ms
        )()
