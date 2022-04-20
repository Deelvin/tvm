import os
import argparse
import locale
import numpy as np

def parse_log(fname):
  out_map = {}
  
  with open(fname, 'rb') as fle:
    data = fle.read()
    data_decoded = data.decode('utf8')
    for line in data_decoded.split('\n'):
      # if line.find('Sum') == 0:
      #   print(line.split())
      if line.find('vm_mod_') == 0 or line.find('VM::') == 0:
        spl = line.split()
        for i in range(len(spl)):
          if spl[i] == 'cpu0':
            val = 0
            if i > 3:
              num = spl[1]+spl[2]
              val = float(num.replace(',', '.'))
            else:
              val = float(spl[1].replace(',', '.'))
            if not spl[0] in out_map.keys():
              out_map[spl[0]] = [val]
            else:
              out_map[spl[0]].append(val)
  dense_val = 0
  matmul_val = 0
  embeds_val = 0
  other_val = 0
  concat_val = 0
  for i, val in out_map.items():
    data = np.array(val)
    mean_v = np.mean(data)
    std_v = np.std(data)
    print(i, "\t,", mean_v, "\t,", std_v)
    if i.find("_nn_dense") != -1:
      dense_val += mean_v
      continue
    if i.find("_nn_batch_matmul") != -1:
      matmul_val += mean_v
      continue
    if i.find("_take_reshape_take") != -1 or i.find("vm_mod_fused_sum") != -1:
      embeds_val += mean_v
      continue
    if i.find("_concatenate") != -1:
      concat_val += mean_v
      continue

    other_val += mean_v
  # print(out_map)
  print("dense\t", dense_val)
  print("matmul\t", matmul_val)
  print("embeds\t", embeds_val)
  print("concat\t", concat_val)
  print("other\t", other_val)

if __name__ == "__main__":

  locale.setlocale(locale.LC_NUMERIC,"ru_RU.utf8")

  parser = argparse.ArgumentParser()
  parser.add_argument("--file", required=True, help="log file with perrformance data", default="default")

  args = parser.parse_args()
  fname = args.file
  if os.path.exists(fname):
    parse_log(fname)
