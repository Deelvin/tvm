import torch
import torch._dynamo as dynamo

import tvm.testing
import tvm.runtime.relax_vm
from tvm.relax.frontend.torch import from_fx

from bert_utils import get_pytorch_model, get_encoded_input


def main():
    torch_model = get_pytorch_model()
    torch_input = get_encoded_input(return_tensors="pt")
    input_info = [(item.shape, str(item.dtype)) for item in get_encoded_input(return_tensors="np").values()]

    graph_model = dynamo.export(torch_model, **torch_input)[0]
    ir_mod: tvm.ir.IRModule = from_fx(graph_model, input_info, unwrap_unit_return_tuple=False)
    ir_mod: tvm.ir.IRModule = tvm.relax.transform.LegalizeOps()(ir_mod)
    executable: tvm.relax.Executable = tvm.relax.build(ir_mod, target=tvm.target.Target("llvm", host="llvm"))
    # vm: tvm.runtime.relax_vm.VirtualMachine = tvm.relax.VirtualMachine(executable, tvm.cpu())


if __name__ == '__main__':
    main()
