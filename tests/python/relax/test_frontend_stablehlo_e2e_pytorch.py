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

import typing

import torch
import torchvision

import torch_mlir
import torch_mlir_e2e_test.configs
import torch_mlir_e2e_test.framework
import torch_mlir_e2e_test.annotations
import torch_mlir_e2e_test.registry
import torch_mlir_e2e_test.reporting
import torch_mlir_e2e_test.stablehlo_backends.linalg_on_tensors
import torch_mlir_e2e_test.stablehlo_backends.abc
import torch_mlir_e2e_test.utils

import mlir.ir
import mlir.dialects

import tvm.testing
import tvm.runtime.relax_vm
from tvm.relax.frontend.stablehlo import from_stablehlo

Invoker = typing.TypeVar("Invoker")


class ResNet18Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.resnet = torchvision.models.resnet18(pretrained=True).train(False)

    @torch_mlir_e2e_test.annotations.export
    @torch_mlir_e2e_test.annotations.annotate_args([
        None,
        ([1, 3, 224, 224], torch.float32, True),
    ])
    def forward(self, x):
        return self.resnet(x)


@torch_mlir_e2e_test.registry.register_test_case(module_factory=lambda: ResNet18Module())
def ResNet18(module, tu: torch_mlir_e2e_test.framework.TestUtils):
    t = tu.rand(1, 3, 224, 224)
    module.forward(t)


def to_relax(model: str) -> tvm.ir.IRModule:
    with mlir.ir.Context() as context:
        mlir.dialects.stablehlo.register_dialect(context)
        m = mlir.ir.Module.parse(model)
    ir_mod = from_stablehlo(m)
    return ir_mod


def register_pytorch_e2e_tests() -> typing.List[torch_mlir_e2e_test.framework.Test]:
    # from torch_mlir_e2e_test.test_suite import conv
    from torch_mlir_e2e_test.test_suite import matmul
    return torch_mlir_e2e_test.registry.GLOBAL_TEST_REGISTRY


def filter_operations(test: torch_mlir_e2e_test.framework.Test) -> bool:
    try:
        program = test.program_factory()
        example_args = torch_mlir_e2e_test.utils.convert_annotations_to_placeholders(program.forward)
        module = torch_mlir.compile(program, example_args, output_type="stablehlo")
        serialized = str(module)
    except torch_mlir.compiler_utils.TorchMlirCompilerError:
        return True

    stop_list = {
        # operations
        "arith.", "tensor.",
        # dialects
        "chlo",
        # dynamic shapes
        # "?",
    }
    for stop in stop_list:
        if stop in serialized:
            return True
    return False


class TVMBackend(torch_mlir_e2e_test.stablehlo_backends.abc.StablehloBackend):
    def compile(self, module: mlir.ir.Module) -> tvm.runtime.relax_vm.VirtualMachine:
        # TODO(agladyshev): need investigate
        #   arg_shape = mlir.ir.ShapedType(arg)
        #         TypeError: __init__(): incompatible constructor arguments. The following argument types are supported:
        #             1. mlir._mlir_libs._mlir.ir.ShapedType(cast_from_type: mlir._mlir_libs._mlir.ir.Type)
        # ir_mod = from_stablehlo(module)

        ir_mod: tvm.ir.IRModule = to_relax(str(module))
        ir_mod: tvm.ir.IRModule = tvm.relax.transform.LegalizeOps()(ir_mod)
        executable: tvm.relax.Executable = tvm.relax.build(ir_mod, target=tvm.target.Target("llvm", host="llvm"))
        vm: tvm.runtime.relax_vm.VirtualMachine = tvm.relax.VirtualMachine(executable, tvm.cpu())
        return vm

    def load(self, artifact: tvm.runtime.relax_vm.VirtualMachine) -> Invoker:
        class Invoker:
            def __init__(self):
                self.vm: tvm.runtime.relax_vm.VirtualMachine = artifact

            def forward(self, *args):
                inputs = [tvm.nd.array(arg) for arg in args]
                outputs = self.vm["main"](*inputs)
                return outputs.numpy()

        return Invoker()


if __name__ == "__main__":
    tests = register_pytorch_e2e_tests()
    tests = [test for test in tests if not filter_operations(test)]

    backend = TVMBackend()
    config = torch_mlir_e2e_test.configs.StablehloBackendTestConfig(backend)
    results = torch_mlir_e2e_test.framework.run_tests(tests, config, sequential=True)
    torch_mlir_e2e_test.reporting.report_results(results, set(), verbose=True)
