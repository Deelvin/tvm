from ast import arg
import octomizer
import requests
import json
import os
from datetime import *
import argparse
import matplotlib.pyplot as plt
import octomizer.client as octoclient
import octomizer.package_type
from octoml.octomizer.v1.workflows_pb2 import WorkflowStatus
from subprocess import Popen, PIPE


def download_tuning_records(workflow, download_path=None):
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

def download_model(model_uuid, system_name):
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
  output_decoded = output.decode('utf8')
  output_to_check = output_decoded.split('\n')
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

# Hack to make it a method.
octomizer.workflow.Workflow.download_tuning_records = download_tuning_records

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--token", required=False, help="Octomizer API token", default="default")
  parser.add_argument("--model-uuid", required=False, help="required model from OctoML.ai", default="default")
  parser.add_argument("--system-name", required=True, help="system name which should be used for data extraction", default="default")
  parser.add_argument("--download-log", required=False, help="workload uuid", default="default")
  parser.add_argument("--model-name", required=True, help="model name for inference [bert, resnet, dlrm]", default="default")
  parser.add_argument("--model-path", required=False, help="model path if log fiel is used for analysis", default="default")
  parser.add_argument("--batch-size", required=False, help="batch size definition for inference, default value is 1", default=1, type=int)
  parser.add_argument("--trial-time", required=False, help="inference time definition in sec, default value is 10", default=10, type=int)
  
  args = parser.parse_args()
  model_name = args.model_name
  model_path = args.model_path
  batch_size_cmd = '--batch-size={}'.format(args.batch_size)
  model_name_cmd = '--model-name={}'.format(model_name)
  trial_time_cmd = '--trial-time={}'.format(args.trial_time)
  if args.token != "default":
    os.environ['OCTOMIZER_API_TOKEN'] = args.token
  client = octoclient.OctomizerClient()
  curr_user = client.get_current_user()
  file_path = os.path.dirname(os.path.realpath(__file__))

  if args.download_log == 'default':
    model_lib_path = os.path.join(file_path, download_model(args.model_uuid, args.system_name))
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
      workflow.download_tuning_records(download_path=log_file)
    except:
      print("ERROR: cannot download required workflow with required id.")
      exit(-1)
    os.chdir(os.path.join(file_path, 'DLRM_mi'))
    cmd_compile = ['python', './compile.py', '--model-path={}'.format(model_path), model_name_cmd, '--tuning-log-file=.{}'.format(log_file), batch_size_cmd]
    p = Popen(cmd_compile, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    output_decoded = output.decode('utf8')
    output_to_check = output_decoded.split('\n')
    model_lib_path = ''
    for line in output_to_check:
      if line.find('Generated library:') != -1:
        model_lib_path = '--model-path={}'.format(line.split()[-1])
    if len(model_path) == 0:
      print("ERROR: model path was not defined.")
      exit(-1)
  cmd = ['python', './multi_instance.py', model_name_cmd, model_lib_path, batch_size_cmd, trial_time_cmd]
  print(cmd)
  p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  output, err = p.communicate()
  os.chdir(file_path)
  if len(output) != 0:
    draw_legend(output, args.system_name, model_name)
  else:
    print("ERROR: empty inference output.")
