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
# if [ $# -ge 1 ] && [[ "$1" = "--device" ]]; then
#     shift 1
#     device_serial="$1"
#     shift
# fi

source tests/scripts/setup-pytest-env.sh
make cython3
echo ICE
rm -rf python/MANIFEST.in python/tvm/microtvm_template_projects/ python/tvm/standalone_crt/ python/tvm/libtvm_runtime.so  python/tvm/libtvm.so configs/

# if [[ "${device_serial}" == "simulator" ]]; then
export TVM_TRACKER_PORT=9190
export TVM_TRACKER_HOST=0.0.0.0
env PYTHONPATH=python python3 -m tvm.exec.rpc_tracker --host "${TVM_TRACKER_HOST}" --port "${TVM_TRACKER_PORT}" &
TRACKER_PID=$!
sleep 5   # Wait for tracker to bind

# Temporary workaround for symbol visibility
export HEXAGON_SHARED_LINK_FLAGS="-Lbuild/hexagon_api_output -lhexagon_rpc_sim"
# fi

# num_of_devices=0
# if [ ! "${device_serial}" == "simulator" ]; then
#     IFS=',' read -ra ADDR <<< "$device_serial"
#     for i in "${ADDR[@]}"; do
#         num_of_devices=$(($num_of_devices+1))
#     done
# fi

export ANDROID_SERIAL_NUMBER=${device_serial}

# pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write
pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write
# pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_cache_read_write
# pytest tests/python/contrib/test_hexagon/test_cache_read_write.py::test_hexa
# pytest tests/python/unittest/test_tir_analysis_calculate_inout_tensors.py::test_hexa
# pytest tests/python/contrib/test_hexagon/test_launcher.py

# run_pytest ctypes python-contrib-hexagon tests/python/contrib/test_hexagon::
# if [ "${device_serial}" == "simulator" ]; then
#     echo ICE
#     run_pytest ctypes python-contrib-hexagon tests/python/contrib/test_hexagon
# else
#     run_pytest ctypes python-contrib-hexagon tests/python/contrib/test_hexagon --tx $num_of_devices*popen --dist=load
# fi

kill ${TRACKER_PID}
# if [[ "${device_serial}" == "simulator" ]]; then
#     kill ${TRACKER_PID}
# fi
