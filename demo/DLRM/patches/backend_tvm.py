"""
TVM native backend for dlrm model
"""
# pylint: disable=unused-argument,missing-docstring
import os
import onnx  # currently supports pytorch1.0
import torch
import backend
# from dlrm_s_pytorch import DLRM_Net
import numpy as np
import tvm
from tvm.contrib import graph_executor

class BackendTVM(backend.Backend):
    def __init__(self):
        super(BackendTVM, self).__init__()
        self.model = None
        self.batch_size = 100
        self.dense = None
        self.ls_o  = np.zeros((26, self.batch_size))
        for i in range(0, self.batch_size):
          self.ls_o[0, i] = i
        for i in range(1, 26):
          self.ls_o[i, :] = self.ls_o[0, :]
        self.ls_i  = None
        self.save_index = 0
        self.log_file = ''
        if 'TVM_LOG_FILE' in os.environ:
            self.log_file = os.environ.get('TVM_LOG_FILE', '')
        else:
            print("Warning 'TVM_LOG_FILE' varible is not set. Default configuration will be used.")
        print("log file: ", self.log_file)
        print("Using CPU...")

    def version(self):
        return tvm.__version__

    def name(self):
        return "tvm-native-dlrm"

    def load(self, model_path, inputs=None, outputs=None):
        shape_dict = {
            "input.1": (self.batch_size, 13),
            "lS_o":    (26, self.batch_size),
            "lS_i":    (26, self.batch_size)
        }
        self.dense = np.zeros((self.batch_size, 13))
        self.ls_i  = np.zeros((26, self.batch_size), dtype=np.int64)
        model_path = self.log_file
        model_json_file = model_path[:-2] + "json"
        model_param_file = model_path[:-2] + "npz"
        with open(model_json_file, "r") as json_f:
            json = json_f.read()

        lib = tvm.runtime.load_module(model_path)
        self.model = graph_executor.create(json, lib, tvm.cpu())
        params = np.load(model_param_file)
        self.model.set_input(**params)
        self.model.set_input("lS_o", self.ls_o.astype(np.int64))
        return self

    def predict(self, batch_dense_X, batch_lS_o, batch_lS_i):
        output = torch.zeros((batch_dense_X.shape[0], 1))
        if self.model:
            for i in range(0, batch_dense_X.shape[0], self.batch_size):
                delta = batch_dense_X.shape[0] - i
                offs = min(i + self.batch_size, batch_dense_X.shape[0])
                delta = batch_dense_X.shape[0] - i
                self.dense[0:delta, :] = batch_dense_X[i:offs, :]
                self.ls_i[:, 0:delta] = batch_lS_i[:, i:offs]
                self.model.set_input("input.1", self.dense)
                self.model.set_input("lS_i", self.ls_i)
                self.model.run()
                x = torch.from_numpy(self.model.get_output(0).asnumpy())
                output[i : min(offs, batch_dense_X.shape[0]) , :] = x[0:min(self.batch_size, delta), :]
        return output
