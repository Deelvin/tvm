import argparse
import os
import tvm
import numpy as np
from tvm.runtime.profiling import Report
from tvm.runtime import profiler_vm

def profile(model_path):
  head, _ = os.path.split(model_path)
  model_param_file = os.path.join(head, "vm_exec_code.ro")
  model_param_file_common = os.path.join(head, "consts")
  code = bytearray(open(model_param_file, "rb").read())
  lib = tvm.runtime.load_module(model_path)
  mod = tvm.runtime.vm.Executable.load_exec(code, lib)
  mod.load_late_bound_consts(model_param_file_common)
  g_mod = profiler_vm.VirtualMachineProfiler(mod, tvm.cpu())
  batch_size = 1
  #resnet check 
  x_in = np.random.randint(0, 100, size=[batch_size, 3, 224, 224]).astype("float32")
  inpt = {"input_tensor:0": x_in}
  for _ in range(10):
    report = g_mod.profile(
        **inpt,
        func_name="main"
    )
    print(report)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model-path", required=True, help="path to the model.", default="default")
  args = parser.parse_args()
  if args.model_path != 'default':
    profile(args.model_path)