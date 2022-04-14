from ast import arg
from email.policy import default
import octomizer
import requests
import json
import os
import onnx
import numpy as np
from datetime import *
import argparse
import matplotlib.pyplot as plt
import octomizer.client as octoclient
import octomizer.project as project
import octomizer.models.onnx_model as onnx_model
import octomizer.package_type
from octoml.octomizer.v1.workflows_pb2 import WorkflowStatus
from octomizer.model_variant import AutoschedulerOptions
from subprocess import Popen, PIPE
from DLRM_mi.multi_instance import main_call as multi_instance_main
from DLRM_mi.models import get_input, Args

PROJECT_NAME = "multi_instance_check"
DESCRIPTION = "Tuning testing project to analize system performance."

SLEEP_TIME = 5 * 60 # 5 minutes
ITERATIONS_COUNT_LIMIT = 120 * 60 // SLEEP_TIME # 2 hours


def download_tuning_records(workflow, client=None, download_path=None):
    assert workflow.proto.status.state == WorkflowStatus.COMPLETED, f"workflow status must be COMPLETED, found {workflow.proto.status}"
    workflow_result = workflow.proto.status.result
    assert workflow_result.HasField("autotune_result"), f"workflow must contain an auto-tuning step"
    octomized_model_variant_uuid =  workflow_result.autotune_result.model_variant_uuid
    octomized_model_variant = client.get_model_variant(uuid=octomized_model_variant_uuid)
    model_format_config = octomized_model_variant.proto.model_format_config
    assert model_format_config.HasField("relay_model_config"), "the auto-tuning workflow must contain a TVM model"
    best_log_dataref_uuid = model_format_config.relay_model_config.best_log_dataref_uuid
    best_log_dataref = client.get_dataref(best_log_dataref_uuid)
    records = requests.get(best_log_dataref.url)
    raw_log = records.content.decode("utf-8")
    if download_path:
        with open(download_path, 'w') as file:
            file.write(raw_log)
    raw_log = "[" + ( ",\n".join(raw_log.split("\n")))[:-2] + "]"
    return json.loads(raw_log)

def upload_workflow_so_and_save(workflow):
  output_filename = workflow.save_package(out_dir=".", package_type=octomizer.package_type.PackageType.LINUX_SHARED_OBJECT)
  p = Popen(['tar', '-xvzf', output_filename], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if p.returncode != 0:
    print("Extraction ERROR!! ", output)
    print("return code ", p.returncode)
    exit(-1)
  output_decoded = output.decode('utf8')
  folder_to_check = output_decoded.split('\n')[0]
  model_name = folder_to_check.split('/')[0]
  lib = os.path.join(folder_to_check, '{}.so'.format(model_name))
  print(lib)
  if os.path.isfile(lib):
    return lib
  return None


def download_model(client, model_uuid, system_name):
  print(dir(client))
  p = Popen(['octomizer', '--json', 'list-workflows', model_uuid], stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  if len(output) == 0 or p.returncode != 0:
    print("Octomizer downloading ERROR!! ", output)
    print("return code ", p.returncode)
    print("error: ", err)
    exit(-1)
  json_decoded = output.decode('utf8').replace("'", '"')
  data = json.loads(json_decoded)
  time_val = None
  data_to_extract = {}

  for i in data:
    # search for the latest Octomizer result
    spec = i['hardware']['platform']
    if spec == system_name:
      tm_val = datetime.strptime(i['createTime'], "%Y-%m-%dT%H:%M:%S.%fZ")
      to_upd = True
      if time_val == None:
        time_val = tm_val
      else:
        if time_val > tm_val:
          to_upd = False
      if i['status']['state'] != 'COMPLETED':
        if to_upd == True:
          to_upd = False # should not extract data

      if to_upd:
        data_to_extract[spec] = i
  for _, val in data_to_extract.items():
    workflow = client.get_workflow(uuid=val['uuid'])
    res = upload_workflow_so_and_save(workflow)
    if res:
      return res
  return None

def shapes_and_types_from_json(file_descriptor):
  with open(file_descriptor, "r") as f:
    loaded_json = json.load(f)
  shape_dict = {}
  inpt_types = {}
  for i in loaded_json:
    xx = loaded_json[i]
    # fields validation
    for key, _ in xx.items():
      if not key in ['input_dtype', 'shape']:
        print("ERROR()")
        exit(0)
    shape_dict[i] = xx['shape']
    inpt_types[i] = xx['input_dtype']
  # print(shape_dict)
  # print(inpt_types)
  return shape_dict, inpt_types

def tune_model_by_octomizer(client, args):
  # print("tune_model_by_octomizer ", args.model_path)
  if not os.path.exists(args.model_path):
    print("ERROR: model {} does not exist.".format(args.model_path))
    return False
  # parse model inputs

  model = onnx.load(args.model_path)
  if args.inputs_descriptor != 'default':
    inpt_shapes, inpt_types = shapes_and_types_from_json(args.inputs_descriptor)
  else:
    inpt_shapes, inpt_types = get_input(model, args.batch_size, args.model_name)
  del model
  # print("tune_model_by_octomizer ", inpt_shapes, inpt_types)
  octo_project = project.Project(
        client,
        name=PROJECT_NAME,
        description=DESCRIPTION,
    )
  if args.num_threads != -1:
    model_package_name = "{}_octomized_thr_{}".format(os.path.basename(args.model_path), args.num_threads)
  else:
    model_package_name = "{}_octomized".format(os.path.basename(args.model_path))
  # print(model_package_name)

  model = onnx_model.ONNXModel(
        client,
        name=model_package_name,
        model=args.model_path,
        description=DESCRIPTION,
        project=octo_project,
    )
  # print("model ", model)
  model_variant = model.get_uploaded_model_variant()
  # print("model_variant ", model_variant)
  if args.num_threads != -1:
    octomize_workflow = model_variant.octomize(
          args.platform,
          tuning_options=AutoschedulerOptions(
              trials_per_kernel=3, early_stopping_threshold=1
          ),
          tvm_num_threads=args.num_threads,
          input_shapes=inpt_shapes,
          input_dtypes=inpt_types
      )
  else:
    octomize_workflow = model_variant.octomize(
          args.platform,
          tuning_options=AutoschedulerOptions(
              trials_per_kernel=3, early_stopping_threshold=1
          ),
          input_shapes=inpt_shapes,
          input_dtypes=inpt_types
      )
  print("Workflow uuid: ", octomize_workflow.uuid)
  print("Please wait notification from Octo.ai and run following commandline:")
  extra_data = ''
  if args.inputs_descriptor != 'default':
    extra_data += "--inputs-descriptor={}".format(args.inputs_descriptor)
  print("python ./run_octomized_model.py --platform={} --workflow-uuid={} --model-name={} --batch-size={} {}".format(args.platform, octomize_workflow.uuid, args.model_name, args.batch_size, extra_data))
  # print("Got into results waiting loop.")
  # # simple solution with requests per 5 minutes to get workflow results.
  # # limited by 2 hours for now
  # exit_from_the_loop = False
  # workflow = None
  # cnt = 0
  # while exit_from_the_loop == False:
  #   timer.sleep(SLEEP_TIME)
  #   workflow_tmp = client.get_workflow(uuid=octomize_workflow.uuid)
  #   if workflow_tmp.status() == 'COMPLETED':
  #     workflow = workflow_tmp
  #     break
  #   cnt += 1
  #   if cnt > ITERATIONS_COUNT_LIMIT:
  #     exit_from_the_loop = True
  # print("Exit from loop.")
  # return upload_workflow_so_and_save(workflow)

def parse_output(lines):
  throughput = []
  latency = []
  lables = []
  max_thr = 0
  m_i_max = 0
  m_t_max = 0

  min_lat = 100000000.
  m_i_min = 0
  m_t_min = 0

  for pos in range(len(lines)):
    line = lines[pos]
    ps = line.find('CFG:')
    if ps != -1:
      if ps == 0:
        num_inst_l = line[ps:-1]
        i_and_t = num_inst_l.split()[0][4:-1].split('/')[0].split('-')
        i_val = int(i_and_t[0])
        t_val = int(i_and_t[1])
        spl = num_inst_l.split('AVG_THR:')
        throughput.append(float(spl[-1]))
        lables.append((i_val, t_val))
        lat = float(num_inst_l.split('AVG_LAT:')[1].split()[0][0:-1])
        latency.append(lat)
        ttt = float(spl[-1])
        if max_thr < ttt:
          max_thr = ttt
          m_i_max = i_val
          m_t_max = t_val
        if min_lat > lat:
          min_lat = lat
          m_i_min = i_val
          m_t_min = t_val

  print("max throughput {}, i:{}, t:{}".format(max_thr, m_i_max, m_t_max))
  print("min latency {}, i:{}, t:{}".format(min_lat, m_i_min, m_t_min))

  return latency, throughput, lables

def draw_legend(output, system_name, model_name):
  with open(output, "r") as fle:
    output_to_check = fle.readlines()

  latency, throughput, labels = parse_output(output_to_check)

  plt.xlabel('latency')
  plt.ylabel('throughput')
  plt.title('{} FP32. {}'.format(model_name, system_name))
  plt.scatter(latency, throughput, marker="o")
  # to search problem areas
  # for v in range(len(labels_1)):
  #   plt.annotate("i{},t{}".format(labels_1[v][0], labels_1[v][1]), (latency_1[v], throughput_1[v]))
  # plt.legend()
  plt.savefig('res_{}_{}.png'.format(system_name, model_name))
  plt.clf()

def gen_numpy_array(input_dtype, shape):
  if input_dtype == 'int64':
    inp_val = np.random.randint(0, 100, size=shape).astype(input_dtype)
  else:
    inp_val = np.random.rand(*shape).astype(input_dtype)
  return inp_val

def generate_inputs(inputs, batch_size):
  res = {}
  for i in inputs.input_fields:
    shape = []
    for j in i.input_shape:
      if j == -1:
        shape.append(batch_size)
      else:
        shape.append(j)
    res[i.input_name] = gen_numpy_array(i.input_dtype, shape)
  return res

def parse_json_inputs(file_descriptor):
  with open(file_descriptor, "r") as f:
    loaded_json = json.load(f)
  print(loaded_json)
  res = {}
  for i in loaded_json:
    xx = loaded_json[i]
    # fields validation
    for key, _ in xx.items():
      if not key in ['input_dtype', 'shape']:
        print("ERROR()")
        exit(0)
      # print(val)
    res[i] = gen_numpy_array(xx['input_dtype'], xx['shape'])
  return res

# Hack to make it a method.
octomizer.workflow.Workflow.download_tuning_records = download_tuning_records

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--token", required=False, help="Octomizer API token", default="default")
  parser.add_argument("--model-uuid", required=False, help="required model from OctoML.ai", default="default")
  parser.add_argument("--platform", required=True, help="platform which should be used for data extraction", default="default")
  parser.add_argument("--download-log", required=False, help="workload uuid", default="default")
  parser.add_argument("--workflow-uuid", required=False, help="workflow uuid to perform measurements.", default="default")
  parser.add_argument("--model-name", required=False, help="model name for inference [bert, resnet, dlrm]", default="default")
  parser.add_argument("--model-path", required=False, help="model path if log file is used for analysis", default="default")
  parser.add_argument("--batch-size", required=False, help="batch size definition for inference, default value is 1", default=1, type=int)
  parser.add_argument("--trial-time", required=False, help="inference time definition in sec, default value is 10", default=10, type=int)
  parser.add_argument("--num-threads", required=False, help="number threads for tuning.", default=-1, type=int)
  parser.add_argument("--output-log", required=False, help="name of output file with latency/throughput points.", default="./output_Log.txt")
  parser.add_argument("--inputs-descriptor", required=False, help="name of json file with inputs description.", default="default")

  args = parser.parse_args()
  model_name = args.model_name
  model_path = args.model_path
  batch_size_cmd = '--batch-size={}'.format(args.batch_size)
  model_name_cmd = '--model-name={}'.format(model_name)
  trial_time_cmd = '--trial-time={}'.format(args.trial_time)
  if args.token != "default":
    os.environ['OCTOMIZER_API_TOKEN'] = args.token
  if args.inputs_descriptor != "default":
    inputs = parse_json_inputs(args.inputs_descriptor)
  client = octoclient.OctomizerClient()
  targets = client.get_hardware_targets()
  platforms = [x.platform for x in targets]
  # for i in platforms:
  #   print(i)
  if not args.platform in platforms:
    print("ERROR: platform is not defined.")
    print("Please select platform parameter from the list:")
    for elem in platforms:
      print(elem)
    exit(0)
  if args.workflow_uuid != "default":
    workflow = client.get_workflow(uuid=args.workflow_uuid)
    model_lib_path = upload_workflow_so_and_save(workflow)
    inputs = generate_inputs(workflow.proto.package_stage_spec.model_inputs, args.batch_size)
  else:
    curr_user = client.get_current_user()
    inputs = None
    file_path = os.path.dirname(os.path.realpath(__file__))
    if args.download_log == 'default':
      if args.model_uuid != 'default':
        models = client.list_models()
        curr_user_models = [mod.proto for mod in models if mod.proto.created_by == curr_user.uuid and args.model_uuid == mod.proto.uuid]
        if len(curr_user_models) != 1:
          print("ERROR: the amount of found models is not 1. The models list size is ", len(curr_user_models))
          exit(0)
        inputs = generate_inputs(curr_user_models[0].inputs, args.batch_size)
        if len(curr_user_models) == 0:
          print("ERROR: incorrect model uuid: ", args.model_uuid)
          print("Avasilable values:")
          for el in curr_user_models:
            print(el.model_uuid)
          exit(0)
        model_lib_path = os.path.join(file_path, download_model(client, args.model_uuid, args.platform))
      else:
        model_lib_path = tune_model_by_octomizer(client, args)
        exit(0)
      os.chdir(os.path.join(file_path, 'DLRM_mi'))
      if model_lib_path != None:
        # just run model
        model_lib_path = '--model-path={}'.format(model_lib_path)
      else:
        print("ERROR: cannot form command line for inference.")
        exit(-1)
    else:
      try:
        workflow = client.get_workflow(uuid=args.download_log)
        log_file = "./{}.json".format(model_name)
        workflow.download_tuning_records(client=client, download_path=log_file)
      except:
        print("ERROR: cannot download required workflow with required id.")
        exit(-1)
      os.chdir(os.path.join(file_path, 'DLRM_mi'))
      cmd_compile = ['python', './compile.py', '--model-path={}'.format(model_path), model_name_cmd, '--tuning-log-file=.{}'.format(log_file), batch_size_cmd]
      p = Popen(cmd_compile, stdin=PIPE, stdout=PIPE, stderr=PIPE)
      output, err = p.communicate()
      output_decoded = output.decode('utf8')
      output_to_check = output_decoded.split('\n')
      os.chdir(file_path)
      model_lib_path = ''
      for line in output_to_check:
        if line.find('Generated library:') != -1:
          model_lib_path = '--model-path={}'.format(line.split()[-1])
      if len(model_path) == 0:
        print("ERROR: model path was not defined.")
        exit(-1)
  args_new = Args(path=model_lib_path, name=model_name, batch_size=args.batch_size, trial_time=args.trial_time)
  args_new.output_log = args.output_log
  res = multi_instance_main(args_new)
  # os.chdir(file_path)
  draw_legend(args_new.output_log, args.platform, model_name)
