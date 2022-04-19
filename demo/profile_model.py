import argparse
import os
import tvm
import numpy as np
from tvm.runtime.profiling import Report
from tvm.runtime import profiler_vm
import octomizer.client as octoclient
from run_octomized_model import generate_inputs, upload_workflow_so_and_save, parse_json_inputs

ITERATIONS_NUMBER = 20

def profile(model_path, inputs):
  head, _ = os.path.split(model_path)
  model_param_file = os.path.join(head, "vm_exec_code.ro")
  model_param_file_common = os.path.join(head, "consts")
  code = bytearray(open(model_param_file, "rb").read())
  lib = tvm.runtime.load_module(model_path)
  mod = tvm.runtime.vm.Executable.load_exec(code, lib)
  mod.load_late_bound_consts(model_param_file_common)
  g_mod = profiler_vm.VirtualMachineProfiler(mod, tvm.cpu())
  for _ in range(ITERATIONS_NUMBER):
    report = g_mod.profile(
        **inputs,
        func_name="main",
        # collectors=[tvm.runtime.profiling.PAPIMetricCollector({tvm.cpu(): ["PAPI_FP_OPS"]})],
    )
    print(report)

def get_workflow_and_inputs_from_workflow_uuid(uuid, batch_size):
  client = octoclient.OctomizerClient()
  workflow = client.get_workflow(uuid=uuid)
  print(workflow)
  inputs = generate_inputs(workflow.proto.package_stage_spec.model_inputs, batch_size)
  return workflow, inputs

def use_workflow(uuid, batch_size):
  workflow, inputs = get_workflow_and_inputs_from_workflow_uuid(uuid, batch_size)
  model_lib_path = upload_workflow_so_and_save(workflow)
  profile(model_lib_path, inputs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--workflow-uuid", required=False, help="workflow uuid to perform measurements.", default="default")
  parser.add_argument("--batch-size", required=False, help="batch size definition for inference, default value is 1", default=1, type=int)
  parser.add_argument("--model-path", required=False, help="path to compiled model", default="default")
  parser.add_argument("--inputs-descriptor", required=False, help="name of json file with inputs description.", default="default")
  args = parser.parse_args()
  if args.model_path != "default":
    if args.workflow_uuid != "default":
      _, inputs = get_workflow_and_inputs_from_workflow_uuid(args.workflow_uuid, args.batch_size)
      profile(args.model_path, inputs)
    else:
      if args.inputs_descriptor != "default":
        inputs = parse_json_inputs(args.inputs_descriptor)
        profile(args.model_path, inputs)
      else:
        print("ERROR: not all requierd parameters were set.")
        exit(0)
      # json should be used for parameters definition
  else:
    if args.workflow_uuid != "default":
      use_workflow(args.workflow_uuid, args.batch_size)
