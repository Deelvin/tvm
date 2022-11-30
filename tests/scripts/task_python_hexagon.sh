#!/usr/bin/env bash
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

set -euxo pipefail

device_serial="simulator"
if [ $# -ge 1 ] && [[ "$1" = "--device" ]]; then
    shift 1
    device_serial="$1"
    shift
fi

source tests/scripts/setup-pytest-env.sh
make cython3

if [[ "${device_serial}" == "simulator" ]]; then
    export TVM_TRACKER_PORT=9190
    export TVM_TRACKER_HOST=0.0.0.0
    env PYTHONPATH=python python3 -m tvm.exec.rpc_tracker --host "${TVM_TRACKER_HOST}" --port "${TVM_TRACKER_PORT}" &
    TRACKER_PID=$!
    sleep 5   # Wait for tracker to bind

    # Temporary workaround for symbol visibility
    export HEXAGON_SHARED_LINK_FLAGS="-Lbuild/hexagon_api_output -lhexagon_rpc_sim"
fi

num_of_devices=0
if [ ! "${device_serial}" == "simulator" ]; then
    IFS=',' read -ra ADDR <<< "$device_serial"
    for i in "${ADDR[@]}"; do
        num_of_devices=$(($num_of_devices+1))
    done
fi

export ANDROID_SERIAL_NUMBER=${device_serial}
if [ "${device_serial}" == "simulator" ]; then
    # pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write
    # pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_vtcm_limit
    # pytest tests/python/contrib/test_hexagon/test_meta_schedule.py::test_dense_relay_auto_schedule
    # run_pytest ctypes python-contrib-hexagon tests/python/contrib/test_hexagon
    # pytest tests/python/contrib/test_hexagon/test_vtcm.py::test_vtcm_limit
    pytest tests/python/unittest/test_runtime_rpc.py::test_rpc_tracker_register
else
    # /git/tvm/tests/python/contrib/test_hexagon/test_vtcm.py
    # pytest tests/python/contrib/test_hexagon/test_vtcm.py::test_vtcm_limit
    # pytest tests/python/unittest/test_runtime_rpc.py::test_rpc_tracker_register
    # pytest tests/python/unittest/test_runtime_rpc.py::test_rpc_tracker_request
    # pytest tests/python/unittest/test_runtime_rpc.py::test_rpc_tracker_via_proxy
    pytest tests/python/contrib/test_hexagon/test_vtcm.py::test_vtcm_limit
    # pytest tests/python/contrib/test_hexagon/test_meta_schedule.py::test_vrmpy_dense
    # pytest tests/python/unittest/test_tir_analysis_calculate_allocated_memory.py::test_matmul_mix_scope
    # python tests/python/contrib/test_rpc_tracker.py
    # python tests/python/all-platform-minimal-test/test_minimal_target_codegen_llvm.py
    # python tests/python/all-platform-minimal-test/test_minimal_target_codegen_llvm.py::test_llvm_add_pipeline
    # pytest tests/python/unittest/test_tir_analysis_calculate_allocated_memory.py::test_vtcm_lowering
    # pytest tests/python/contrib/test_hexagon/test_meta_schedule.py::test_dense_relay_auto_schedule
    # run_pytest ctypes python-contrib-hexagon tests/python/contrib/test_hexagon -n=$num_of_devices
    # pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write
fi

if [[ "${device_serial}" == "simulator" ]]; then
    kill ${TRACKER_PID}
fi
