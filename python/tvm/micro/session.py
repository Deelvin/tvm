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

"""Defines a top-level glue class that operates the Transport and Flasher classes."""

import json
import logging
import sys

from ..error import register_error
from .._ffi import get_global_func, register_func
from ..contrib import graph_executor
from ..contrib import utils
from ..contrib.debugger import debug_executor
from ..rpc import RPCSession
from . import project
from .transport import IoTimeoutError
from .transport import TransportLogger

try:
    from .base import _rpc_connect
except ImportError:
    raise ImportError("micro tvm is not enabled. Set USE_MICRO to ON in config.cmake")


@register_error
class SessionTerminatedError(Exception):
    """Raised when a transport read operationd discovers that the remote session is terminated."""


class Session:
    """MicroTVM Device Session

    Parameters
    ----------
    config : dict
        configuration for this session (as generated by
        `tvm.micro.device.host.default_config()`, for example)

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      dev_config = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
      with tvm.micro.Session(dev_config) as sess:
          micro_mod = sess.create_micro_mod(c_mod)
    """

    def __init__(
        self,
        transport_context_manager=None,
        session_name="micro-rpc",
        timeout_override=None,
    ):
        """Configure a new session.

        Parameters
        ----------
        transport_context_manager : ContextManager[transport.Transport]
            If given, `flasher` and `binary` should not be given. On entry, this context manager
            should establish a transport between this TVM instance and the device.
        session_name : str
            Name of the session, used for debugging.
        timeout_override : TransportTimeouts
            If given, TransportTimeouts that govern the way Receive() behaves. If not given, this is
            determined by calling has_flow_control() on the transport.
        """
        self.transport_context_manager = transport_context_manager
        self.session_name = session_name
        self.timeout_override = timeout_override

        self._rpc = None
        self._graph_executor = None

        self._exit_called = False

    def get_system_lib(self):
        return self._rpc.get_function("runtime.SystemLib")()

    def _wrap_transport_read(self, n, timeout_microsec):
        try:
            return self.transport.read(
                n, float(timeout_microsec) / 1e6 if timeout_microsec is not None else None
            )
        except IoTimeoutError:
            return bytes([])

    def _wrap_transport_write(self, data, timeout_microsec):
        self.transport.write(
            data, float(timeout_microsec) / 1e6 if timeout_microsec is not None else None
        )

        return len(data)  # TODO(areusch): delete

    def __enter__(self):
        """Initialize this session and establish an RPC session with the on-device RPC server.

        Returns
        -------
        Session :
            Returns self.
        """
        self.transport = TransportLogger(
            self.session_name, self.transport_context_manager, level=logging.DEBUG
        ).__enter__()

        try:
            timeouts = self.timeout_override
            if timeouts is None:
                timeouts = self.transport.timeouts()

            self._rpc = RPCSession(
                _rpc_connect(
                    self.session_name,
                    self._wrap_transport_write,
                    self._wrap_transport_read,
                    int(timeouts.session_start_retry_timeout_sec * 1e6),
                    int(timeouts.session_start_timeout_sec * 1e6),
                    int(timeouts.session_established_timeout_sec * 1e6),
                    self._cleanup,
                )
            )
            self.device = self._rpc.cpu(0)
            return self

        except:
            self.transport.__exit__(*sys.exc_info())
            raise

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Tear down this session and associated RPC session resources."""
        if not self._exit_called:
            self._exit_called = True
            self.transport.__exit__(exc_type, exc_value, exc_traceback)

    def _cleanup(self):
        self.__exit__(None, None, None)


def lookup_remote_linked_param(mod, storage_id, template_tensor, device):
    """Lookup a parameter that has been pre-linked into a remote (i.e. over RPC) Module.

    This function signature matches the signature built by

    Parameters
    ----------
    mod : tvm.runtime.Module
        The remote Module containing the pre-linked parameters.
    storage_id : int
        An integer identifying the pre-linked paramter to find
    template_tensor : DLTensor
        A DLTensor containing metadata that should be filled-in to the returned NDArray. This
        function should mostly not inspect this, and just pass it along to
        NDArrayFromRemoteOpaqueHandle.
    device : Device
        The remote CPU device to be used with the returned NDArray.

    Returns
    -------
    tvm.nd.NDArray :
        NDArray containing the pre-linked parameter.
    """
    try:
        lookup_linked_param = mod.get_function("_lookup_linked_param")
    except AttributeError:
        return None

    remote_data = lookup_linked_param(storage_id)
    if remote_data is None:
        return None

    return get_global_func("tvm.rpc.NDArrayFromRemoteOpaqueHandle")(
        mod, remote_data, template_tensor, device, None
    )


def create_local_graph_executor(graph_json_str, mod, device):
    """Create a local graph executor driving execution on the remote CPU device given.

    Parameters
    ----------
    graph_json_str : str
        A string containing the graph representation.

    mod : tvm.runtime.Module
        The remote module containing functions in graph_json_str.

    device : tvm.runtime.Device
        The remote CPU execution device.

    Returns
    -------
    tvm.contrib.GraphExecutor :
         A local graph executor instance that executes on the remote device.
    """
    device_type_id = [device.device_type, device.device_id]
    fcreate = get_global_func("tvm.graph_executor.create")
    return graph_executor.GraphModule(
        fcreate(graph_json_str, mod, lookup_remote_linked_param, *device_type_id)
    )


def create_local_debug_executor(graph_json_str, mod, device, dump_root=None):
    """Create a local debug runtime driving execution on the remote CPU device given.

    Parameters
    ----------
    graph_json_str : str
        A string containing the graph representation.

    mod : tvm.runtime.Module
        The remote module containing functions in graph_json_str.

    device : tvm.runtime.Device
        The remote CPU execution device.

    dump_root : Optional[str]
        If given, passed as dump_root= to GraphModuleDebug.

    Returns
    -------
    tvm.contrib.GraphExecutor :
         A local graph executor instance that executes on the remote device.
    """
    device_type_id = [device.device_type, device.device_id]
    fcreate = get_global_func("tvm.graph_executor_debug.create")
    return debug_executor.GraphModuleDebug(
        fcreate(graph_json_str, mod, lookup_remote_linked_param, *device_type_id),
        [device],
        graph_json_str,
        dump_root=dump_root,
    )


@register_func("tvm.micro.compile_and_create_micro_session")
def compile_and_create_micro_session(
    mod_src_bytes: bytes,
    template_project_dir: str,
    project_options: dict = None,
):
    """Compile the given libraries and sources into a MicroBinary, then invoke create_micro_session.

    Parameters
    ----------
    mod_src_bytes : bytes
        The content of a tarfile which contains the TVM-generated sources which together form the
        SystemLib. This tar is expected to be created by export_library. The tar will be extracted
        into a directory and the sources compiled into a MicroLibrary using the Compiler.

    template_project_dir: str
        The path to a template microTVM Project API project which is used to generate the embedded
        project that is built and flashed onto the target device.

    project_options: dict
        Options for the microTVM API Server contained in template_project_dir.
    """

    temp_dir = utils.tempdir()
    # Keep temp directory for generate project
    temp_dir.set_keep_for_debug(True)
    model_library_format_path = temp_dir / "model.tar.gz"
    with open(model_library_format_path, "wb") as mlf_f:
        mlf_f.write(mod_src_bytes)

    try:
        template_project = project.TemplateProject.from_directory(template_project_dir)
        generated_project = template_project.generate_project_from_mlf(
            model_library_format_path,
            str(temp_dir / "generated-project"),
            options=json.loads(project_options),
        )
    except Exception as exception:
        logging.error("Project Generate Error: %s", str(exception))
        raise exception

    generated_project.build()
    generated_project.flash()
    transport = generated_project.transport()

    rpc_session = Session(transport_context_manager=transport)
    # RPC exit is called by cleanup function.
    rpc_session.__enter__()
    return rpc_session._rpc._sess
