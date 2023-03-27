import argparse

import tvm
from tvm import relay, relax, transform
from tvm.contrib.download import download_testdata
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.testing.relay_translator import from_relay
from tvm.ir.module import IRModule
from tvm.relax.testing import relay_translator

import cv2
import onnx
import numpy as np

from tvm.target import Target

SSD_MOBILENET_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
FASTER_RCNN_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx"
MASK_RCNN_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx"
YOLO_V3_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx"
TINY_YOLO_V3_URL = "https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"

MODEL_URL_COLLECTION = {
    "ssd": SSD_MOBILENET_URL,   # SSD MobileNetv1
    "fast": FASTER_RCNN_URL,    # Faster-RCNN
    "mask": MASK_RCNN_URL,      # MASK-RCNN
    "yolo": YOLO_V3_URL,        # "YOLO-v3"
    "tiny": TINY_YOLO_V3_URL,   # "Tiny YOLO-v3"
}

target_h = "llvm"

def download_onnx_model(model_name):
    model_url = MODEL_URL_COLLECTION[model_name]

    print("Downloading model...")
    model_file_name = model_url[model_url.rfind("/") + 1:].strip()
    file_name = download_testdata(model_url, model_file_name, module="models")
    print("Model was saved in", file_name)
    print("Loading model...")
    onnx_model = onnx.load(file_name)
    return onnx_model


def load_from_onnx_model(onnx_model, shape_dict):
    print("Importing to TVM...")
    model, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)
    return model, params


def apply_opt_before_tuning(relay_mod, params, target):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        relay_mod = relay.transform.FoldConstant()(relay_mod)

        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
    return relax_mod


def preprocessing(in_size=224, batch_size=1, expand=False):
    img_url = (
        "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
    )
    img_path = download_testdata(img_url, "test_street_small.jpg", module="data")

    img = cv2.imread(img_path).astype("float32")
    img = cv2.resize(img, (in_size, in_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img / 255.0, [2, 0, 1])
    if batch_size > 1 or expand:
        img = np.expand_dims(img, axis=0)
        broadcast_shape = list(img.shape)
        broadcast_shape[0] = batch_size
        img = np.broadcast_to(img, broadcast_shape)
    return img


def get_relax_executor(tvm_model, target, dev):
    # Compile the relax graph into a VM then run.
    with tvm.transform.PassContext(opt_level=3):
        lib = relax.build(tvm_model, target=target)
        exec = relax.VirtualMachine(lib, dev)
    return exec


def main():
    model_list_str = ""
    for model_name in MODEL_URL_COLLECTION.keys():
        model_list_str += " " + model_name + ","
    model_list_str = model_list_str[:-1]

    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Debugging test for some models from https://github.com/onnx/models with dynamism using Relax:" + model_list_str,
        formatter_class=MyFormatter
    )
    # Model format
    parser.add_argument("-m", "--model_name", default="fast", type=str, help=\
        "Model name: 'ssd' for SSD witn MobileNetv1, 'fast' for Faster-RCNN, 'mask' for Mask-RCNN")
    parser.add_argument("-t", "--target", default="opencl", type=str, help=\
        "Target from the list ('opencl', 'cuda', 'llvm')")
    parser.add_argument("-s", "--in_size", default=224, type=int, help=\
        "Size for input image resizing")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help=\
        "Batch size. The same image is used with broadcasting if the batch size is bigger than 1")
    parser.add_argument("-l", "--with_nhwc", action="store_true", help=\
        "Use NHWC layout instead of NCHW. It needs for MobileNetv1-SSD")
    parser.add_argument("-r", "--from_relay", action="store_true", help=\
        "Use method from_relay to extract Relax IR from ONNX model using Relay front-end")

    args = parser.parse_args()

    target_c = args.target
    target = Target(target_c, host=target_h)
    img = preprocessing(args.in_size, args.batch_size, args.model_name in ["ssd", "yolo", "tiny"])
    onnx_model = download_onnx_model(args.model_name)

    shape_dict = {}
    input_dict = {}
    if args.model_name in ["yolo", "tiny"]:
        shape_dict = {
            "image_shape": [1, 2],
            "input_1": img.shape,
        }
        input_dict = {
            "image_shape": tvm.nd.array(np.array(img.shape).astype("float32")[2:]),
            "input_1": tvm.nd.array(img),
        }
    else:
        input_name = ""
        if args.model_name == "mask" or args.model_name == "fast":
            input_name = "image"
        elif args.model_name == "ssd":
            input_name = "image_tensor:0"
            if args.with_nhwc:
                img = np.transpose(img, [0, 2, 3, 1])

        shape_dict[input_name] = img.shape
        input_dict = {input_name: tvm.nd.array(img)}
    if args.from_relay:
        model, params = load_from_onnx_model(onnx_model, shape_dict)
        tvm_model = apply_opt_before_tuning(model, params, target)
    else:
        tvm_model = from_onnx(onnx_model)
    # Legalize any relax ops into tensorir.
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    print("=" * 10)
    tvm_model.show()
    print("=" * 10)

    if target_c == "cuda":
        dev = tvm.cuda()
    elif target_c == "opencl":
        dev = tvm.cl()
    else:
        dev = tvm.cpu()

    exec = get_relax_executor(tvm_model, target, dev)
    exec.set_input("main", **input_dict)
    print("Run...")
    exec.invoke_stateful("main")
    tvm_res = exec.get_outputs("main")
    print("Output...")
    print(tvm_res.numpy())


if __name__ == '__main__':
    main()
