import argparse

import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib.download import download_testdata

import cv2
import onnx
import numpy as np

from tvm.target import Target

SSD_MOBILENET_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
FASTER_RCNN_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx"
MASK_RCNN_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx"

target_h = "llvm"

def download_onnx_model(model_name):
    model_url = None
    if model_name == "ssd":
      model_url = SSD_MOBILENET_URL
    elif model_name == "fast":
      model_url = FASTER_RCNN_URL
    elif model_name == "mask":
      model_url = MASK_RCNN_URL
    else:
      raise ValueError(f"Model name {model_name} is not supported")

    print("Downloading model...")
    model_file_name = model_url[model_url.rfind("/") + 1:].strip()
    file_name = download_testdata(model_url, model_file_name, module="models")
    print("Loading model...")
    onnx_model = onnx.load(file_name)
    return onnx_model


#def create_lib():
#    name = args.input_model
#    onnx_model = onnx.load(name)
#    shape_dict = {}
#    #shape_dict["input"] = [relay.Any(), 3, 224, 224]
#    shape_dict["input"] = [1, 3, 224, 224]
#    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
#    target = Target(target_c, host=target_h)
#    #model, params = relay.frontend.from_onnx(onnx_model, freeze_params=True)
#
#    vmc = relay.backend.vm.compile(model, target=target, params=params) 
#    vmc.mod.export_library(f"{name}.vm.so")
#    text_file = open(f"{name}.vm.json", "w")
#    text_file.write(vmc.bytecode)
#    text_file.close()
#
#    code, lib = vmc.save()
#    lib.export_library(f"{name}.vm.kernels.so")
#    vmc.move_late_bound_consts(f"{name}.vm.const", byte_limit=256)
#    with open(f"{name}.vm.ro", "wb") as fo:
#        fo.write(code)


def load_from_onnx_model(onnx_model, shape_dict):
    print("Importing to TVM...")
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    print("=" * 10)
    print(model)
    print("=" * 10)
    return model, params


def preprocessing(in_size=224):
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
    )
    img_path = download_testdata(img_url, "test_street_small.jpg", module="data")

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    # img = np.expand_dims(img, axis=0)
    return tvm.nd.array(img)


def compile_model(model, params, target, exec_type="vm"):
    exec=None
    if exec_type == "vm":
        with tvm.transform.PassContext(opt_level=3):
            exec = relay.vm.compile(model, target=target, params=params)
    elif exec_type == "graph":
        with tvm.transform.PassContext(opt_level=3):
            exec = relay.build(model, target=target, params=params)
    else:
        raise ValueError("ERROR: Executor type {} is unsupported. ".format(exec_type),
            "Only \"vm\" and \"graph\" types are supported")
    return exec


def get_executor(model, params, target, dev, exec_type="vm"):
    lib = compile_model(model, params, target, exec_type)
    exec=None
    if exec_type == "vm":
        exec = VirtualMachine(lib, dev)
    elif exec_type == "graph":
        exec = GraphModule(lib["default"](dev))
    else:
        raise ValueError("ERROR: Executor type {} is unsupported. ".format(exec_type),
            "Only \"vm\" and \"graph\" types are supported")
    return exec


def set_input(mod, tvm_inputs, exec_type="vm"):
    if exec_type == "vm":
        mod.set_input(func_name="main", **tvm_inputs)
    else:
        mod.set_input(**tvm_inputs)


def main():
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Debugging test for some models from https://github.com/onnx/models with dynamism",
        formatter_class=MyFormatter
    )
    # Model format
    parser.add_argument("-m", "--model_name", default="fast", type=str, help=\
        "Model name: 'ssd' for SSD witn MobileNetv1, 'fast' for Faster-RCNN, 'mask' for Mask-RCNN")
    parser.add_argument("-ex", "--executor", default="vm", type=str, help=\
        "Executor type used by TVM for model compilation. There are two types supported: 'vm' and 'graph'")
    parser.add_argument("-t", "--target", default="opencl", type=str, help=\
        "Target from the list ('opencl', 'cuda', 'llvm')")
    parser.add_argument("-s", "--in_size", default=224, type=int, help=\
        "Size for input image resizing")

    args = parser.parse_args()

    target_c = args.target
    target = Target(target_c, host=target_h)
    img = preprocessing(args.in_size)
    onnx_model = download_onnx_model(args.model_name)

    input_name = ""
    if args.model_name == "mask" or args.model_name == "fast":
        input_name = "image"
    elif args.model_name == "ssd":
        input_name = "image_tensor:0"
    input_dict = {input_name: img}

    shape_dict = {}
    shape_dict[input_name] = [3, args.in_size, args.in_size]
    model, params = load_from_onnx_model(onnx_model, shape_dict)

    if target_c == "cuda":
        dev = tvm.cuda()
    elif target_c == "opencl":
        dev = tvm.cl()
    else:
        dev = tvm.cpu()

    print(f"Create {args.executor} executor...")
    exec = get_executor(model, params, target, dev, args.executor)
    set_input(exec, input_dict, args.executor)
    print("Run...")
    tvm_res = exec.run()
    print("Output...")
    print(tvm_res)


if __name__ == '__main__':
    main()
