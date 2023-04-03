# For GPT-2 only
from transformers import GPT2Tokenizer
import argparse

import tvm
from tvm import relay, relax, transform
from tvm.contrib.download import download_testdata
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.testing.relay_translator import from_relay
from tvm.ir.module import IRModule

import cv2
import onnx
from onnx import mapping
import numpy as np

from tvm.target import Target


target_h = "llvm"

ONNX_MODEL_ZOO_ROOT_URL = "https://github.com/onnx/models/raw/main/"

MOBILENET_2_URL = "vision/classification/mobilenet/model/mobilenetv2-7.onnx"
RESNET50_1_URL = "vision/classification/resnet/model/resnet50-v1-7.onnx"
RESNET50_2_URL = "vision/classification/resnet/model/resnet50-v2-7.onnx"
RESNET152_1_URL = "vision/classification/resnet/model/resnet152-v1-7.onnx"
SQUEEZENET_1_1_URL = "vision/classification/squeezenet/model/squeezenet1.1-7.onnx"
SQUEEZENET_1_0_URL = "vision/classification/squeezenet/model/squeezenet1.0-7.onnx"
INCEPTION_1_URL = "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.onnx"
INCEPTION_2_URL = "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.onnx"
VGG_16_URL = "vision/classification/vgg/model/vgg16-7.onnx"
VGG_19_URL = "vision/classification/vgg/model/vgg19-7.onnx"
DENCENET_121_URL = "vision/classification/densenet-121/model/densenet-9.onnx"
ALEXNET_URL = "vision/classification/alexnet/model/bvlcalexnet-9.onnx"
GOOGLENET_URL = "vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx"
SHUFFLENET_1_URL = "vision/classification/shufflenet/model/shufflenet-9.onnx"
SHUFFLENET_2_URL = "vision/classification/shufflenet/model/shufflenet-v2-12.onnx"
ZFNET_URL = "vision/classification/zfnet-512/model/zfnet512-9.onnx"
CAFFENET = "vision/classification/caffenet/model/caffenet-9.onnx"
RCNN_ILSVRC13_URL = "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx"
EFFICIENTNET_LITE4_URL = "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx"

MNIST_URL = "vision/classification/mnist/model/mnist-8.onnx"

SSD_URL = "vision/object_detection_segmentation/ssd/model/ssd-10.onnx"
SSD_MOBILENET_URL = "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx"
FASTER_RCNN_URL = "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-12.onnx"
MASK_RCNN_URL = "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx"
RETINANET_URL = "vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx"
YOLO_V2_URL = "vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx"
TINY_YOLO_V2_URL = "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx"
YOLO_V3_URL = "vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx"
TINY_YOLO_V3_URL = "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx"
YOLO_V4_URL = "vision/object_detection_segmentation/yolov4/model/yolov4.onnx"
DUC_URL = "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx"
FCN_50_URL = "vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx"
FCN_101_URL = "vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.onnx"

ARC_FACE_URL = "vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx"
ULTRA_FACE_URL_320 = "vision/body_analysis/ultraface/models/version-RFB-320.onnx"
ULTRA_FACE_URL_640 = "vision/body_analysis/ultraface/models/version-RFB-640.onnx"
EMOTION_PLUS_URL = "vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
GOOGLENET_AGE_URL = "vision/body_analysis/age_gender/models/age_googlenet.onnx"
GOOGLENET_GENDER_URL = "vision/body_analysis/age_gender/models/gender_googlenet.onnx"
VGG_ILSVRC16_AGE_URL ="vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx"
VGG_ILSVRC16_GENDER_URL = "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx"

SUPER_RESOLUTION_URL = "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx"
FNST_MOSAIC_URL = "vision/style_transfer/fast_neural_style/model/mosaic-9.onnx"
FNST_CANDY_URL = "vision/style_transfer/fast_neural_style/model/candy-9.onnx"
FNST_RAIN_PRINCESS_URL = "vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx"
FNST_UDNIE_URL = "vision/style_transfer/fast_neural_style/model/udnie-9.onnx"
FNST_POINTILISM_URL = "vision/style_transfer/fast_neural_style/model/pointilism-9.onnx"

BIDAF_URL = "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx"
BERT_URL = "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx"
ROBERTA_BASE_URL = "text/machine_comprehension/roberta/model/roberta-base-11.onnx"
ROBERTA_SEQ_URL = "text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx"
GPT_2_URL = "text/machine_comprehension/gpt-2/model/gpt2-10.onnx"
T5_URL = "text/machine_comprehension/t5/model/t5-encoder-12.onnx"

MODEL_URL_COLLECTION = {
# CV classification
    "mbn-2": MOBILENET_2_URL,       # MobileNet-v2
    "rsn50": RESNET50_1_URL,        # ResNet50-v1
    "rsn50-2": RESNET50_2_URL,      # ResNet50-v2
    "rsn152": RESNET152_1_URL,      # ResNet152-v1
    "sqz-1.1": SQUEEZENET_1_1_URL,  # SqueezeNet-v1.1
    "sqz-1.0": SQUEEZENET_1_0_URL,  # SqueezeNet-v1.0
    "inc": INCEPTION_1_URL,         # Inception-v1
    "inc-2": INCEPTION_2_URL,       # Inception-v2
    "vgg16": VGG_16_URL,            # VGG-16
    "vgg19": VGG_19_URL,            # VGG-19
    "dense": DENCENET_121_URL,      # DenseNet-121
    "alex": ALEXNET_URL,            # AlexNet
    "google": GOOGLENET_URL,        # GoogleNet
    "shuffle": SHUFFLENET_1_URL ,   # ShuffleNet-v1
    "shuffle-2": SHUFFLENET_2_URL,  # ShuffleNet-v2
    "zf": ZFNET_URL,                # ZFNet
    "caffe": CAFFENET,              # CaffeNet
    "ilsvrc13": RCNN_ILSVRC13_URL,  # RCNN_ILSVRC13
    "eff": EFFICIENTNET_LITE4_URL,  # EfficientNet-Lite4
# Handwritten Digit Recognition
    "mnist": MNIST_URL,             # MNIST
# Detection and segmentation
    "ssd": SSD_URL,                 # SSD
    "ssd-mob": SSD_MOBILENET_URL,   # SSD MobileNetv1
    "fast": FASTER_RCNN_URL,        # Faster-RCNN
    "mask": MASK_RCNN_URL,          # MASK-RCNN
    "retina": RETINANET_URL,        # RetinaNet
    "tiny-yolo2": TINY_YOLO_V2_URL, # Tiny YOLO-v2
    "yolo2": YOLO_V2_URL,           # YOLO-v2
    # There is difference with YOLO-v3 from MXNet. It has dynamic shape inside and is not supported by TVM
    "yolo3": YOLO_V3_URL,           # "YOLO-v3"
    "tiny-yolo3": TINY_YOLO_V3_URL, # "Tiny YOLO-v3"
    "yolo4": YOLO_V4_URL,           # YOLO-v4
    "duc": DUC_URL,                 # DUC
    "fcn-50": FCN_50_URL,           # FCN-50
    "fcn-101": FCN_101_URL,         # FCN-101
# Body, Face and Gesture analysis
    "arc": ARC_FACE_URL,            # ArcFace
    "ultra-320": ULTRA_FACE_URL_320,# UltraFace-320
    "ultra-640": ULTRA_FACE_URL_640,# UltraFace-640
    "emotion": EMOTION_PLUS_URL,    # EmotionFerPlus
    "ggl-age": GOOGLENET_AGE_URL,   # GoogleNet-age
    "ggl-gender": GOOGLENET_GENDER_URL,             # GoogleNet-gender
    "vgg_ilsvrc_16_age": VGG_ILSVRC16_AGE_URL,      # VGG_ILSVRC_16-age
    "vgg_ilsvrc_16_gender": VGG_ILSVRC16_GENDER_URL,# VGG_ILSVRC_16-gender
# Image manipulation
    "super_res": SUPER_RESOLUTION_URL,              # SuperResolution
    # Fast Neural Style Transfer. TODO(vvchernov): accuracy test failed (rtol~0,5%)
    "fnst-m": FNST_MOSAIC_URL,      # FNST-mosaic
    "fnst-c": FNST_CANDY_URL,       # FNST-candy
    "fnst-rp": FNST_RAIN_PRINCESS_URL,              # FNST-rain-princess
    "fnst-u": FNST_UDNIE_URL,       # FNST-udnie
    "fnst-p": FNST_POINTILISM_URL,  # FNST-pointilism
# Machine compression
    "bidaf": BIDAF_URL,             # BiDAF
    "bert": BERT_URL,               # BERT
    "roberta-b": ROBERTA_BASE_URL,  # RoBERTa-base
    "roberta-s": ROBERTA_SEQ_URL,   # RoBERTa-seq
    "gpt-2": GPT_2_URL,             # GPT-2
    "t5": T5_URL,                   # T5
}

specific_model_names = [
    "tiny-yolo3",
    "fcn-50",
    "fcn-101",
    "roberta-b",
    "roberta-s",
    "t5",
]


def get_FCN50_shape_dict():
    return {"input": [1, 3, 480, 640]}


def get_FCN101_shape_dict():
    return {"input": [1, 3, 480, 640]}


def get_RoBERTa_shape_dict():
    return {"input_ids": [1, 100]}


def get_T5_shape_dict():
    return {"input_ids": [1, 100]}


def get_TinyYOLOv3_shape_dict():
    return {"input_1": [1, 3, 416, 416],
            "image_shape": [1, 2]}


def get_specific_input_shapes(model_name):
    shapes_dict = {}
    if model_name == "fcn-50":
        shapes_dict = get_FCN50_shape_dict()
    elif model_name == "fcn-101":
        shapes_dict = get_FCN101_shape_dict()
    elif model_name == "roberta-b" or model_name == "roberta-s":
        shapes_dict = get_RoBERTa_shape_dict()
    elif model_name == "t5":
        shapes_dict = get_T5_shape_dict()
    elif model_name == "tiny-yolo3":
        shapes_dict = get_TinyYOLOv3_shape_dict()
    return [shapes_dict[name] for name in sorted(shapes_dict.keys())]


def download_onnx_model(model_name):
    model_url = ONNX_MODEL_ZOO_ROOT_URL + MODEL_URL_COLLECTION[model_name]

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

        relax_mod = from_relay(relay_mod["main"], target=target)
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


def get_onnx_input_names(model):
    inputs = [node.name for node in model.graph.input]
    initializer = [node.name for node in model.graph.initializer]
    inputs = list(set(inputs) - set(initializer))
    return sorted(inputs)


def get_onnx_input_types(model):
    input_names = get_onnx_input_names(model)
    return [
        mapping.TENSOR_TYPE_TO_NP_TYPE[node.type.tensor_type.elem_type]
        for node in sorted(model.graph.input, key=lambda node: node.name) if node.name in input_names
    ]


def get_common_input_shapes(model):
    input_names = get_onnx_input_names(model)
    shapes = [
        [dv.dim_value for dv in node.type.tensor_type.shape.dim]
        for node in sorted(model.graph.input, key=lambda node: node.name) if node.name in input_names
    ]
    for shape in shapes:
        for i in range(len(shape)):
            if shape[i] < 1:
                print("WARNING: dimension of shape is non-positive. It is replaced by 1!")
                shape[i] = 1
    return shapes


def get_onnx_input_shapes(model, model_name = ""):
    if model_name in specific_model_names:
        return get_specific_input_shapes(model_name)
    else:
        return get_common_input_shapes(model)


def get_random_model_inputs(model, model_name = ""):
    inputs = []
    if model_name == "gpt-2":
        # TODO: looks like ORT catches shape mismatch if number of words in the input string is not equal to 8. Pure TVM works fine for any number of words
        test_string = "One two three four five six seven eight"
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        inputs = [np.array([[tokenizer.encode(test_string)]])]
        input_shapes = [inputs[0].shape]
    else:
        input_shapes = get_onnx_input_shapes(model, model_name)
        input_types = get_onnx_input_types(model)
        assert len(input_types) == len(input_shapes)
        inputs = []
        for shape, dtype in zip(input_shapes, input_types):
            high_val = 1.0
            if (dtype == "int32" or dtype == "int64"):
                high_val = 1000.0
            inputs.append(np.random.uniform(size=shape, high=high_val).astype(dtype))
    print("INPUT SHAPES:", input_shapes)
    return inputs, input_shapes


def get_inputs_shapes(onnx_model, model_name, in_size, batch_size, with_nhwc=False, use_image=True):
    shape_dict = {}
    input_dict = {}
    if use_image:
        img = preprocessing(in_size, batch_size, model_name in ["ssd", "yolo", "tiny"])
        if model_name in ["yolo", "tiny"]:
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
            if model_name == "mask" or model_name == "fast":
                input_name = "image"
            elif model_name == "ssd":
                input_name = "image_tensor:0"
                if with_nhwc:
                    img = np.transpose(img, [0, 2, 3, 1])

            shape_dict[input_name] = img.shape
            input_dict = {input_name: tvm.nd.array(img)}
    else:
        inputs, input_shapes = get_random_model_inputs(onnx_model, model_name)
        input_names = sorted(get_onnx_input_names(onnx_model))
        print("INPUT NAMES:", input_names)
        for key, value in zip(input_names, input_shapes):
            shape_dict[key] = value
        for key, value in zip(input_names, inputs):
            input_dict[key] = tvm.nd.array(value)

    return input_dict, shape_dict


def main():
    model_list_str = ""
    for model_name in MODEL_URL_COLLECTION.keys():
        model_list_str += " " + model_name + ","
    model_list_str = model_list_str[:-1]

    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description="Debugging test for some models from https://github.com/onnx/models using Relax:" + model_list_str,
        formatter_class=MyFormatter
    )
    # Model format
    parser.add_argument("-m", "--model_name", default="rsn50", type=str, help=\
        "Shortened model name from ONNX model zoo")
    parser.add_argument("-t", "--target", default="llvm", type=str, help=\
        "Target from the list ('opencl', 'cuda', 'llvm')")
    parser.add_argument("-s", "--in_size", default=224, type=int, help=\
        "Size for input image resizing")
    parser.add_argument("-b", "--batch_size", default=1, type=int, help=\
        "Batch size. The same image is used with broadcasting if the batch size is bigger than 1")
    parser.add_argument("-l", "--with_nhwc", action="store_true", help=\
        "Use NHWC layout instead of NCHW. It needs for MobileNetv1-SSD")
    parser.add_argument("-r", "--from_relay", action="store_true", help=\
        "Use method from_relay to extract Relax IR from ONNX model using Relay front-end")
    parser.add_argument("-i", "--use_image", action="store_true", help=\
        "Test model use real image otherwise it generates random tensor of corresponding size")
    parser.add_argument("-p", "--print", action="store_true", help=\
        "Print the relax model and output tensors")

    args = parser.parse_args()

    target_c = args.target
    target = Target(target_c, host=target_h)
    print("Trying to check model:", args.model_name)
    onnx_model = download_onnx_model(args.model_name)

    input_dict, shape_dict = get_inputs_shapes(
        onnx_model, args.model_name, args.in_size, args.batch_size, args.with_nhwc, args.use_image
    )

    if args.from_relay:
        model, params = load_from_onnx_model(onnx_model, shape_dict)
        tvm_model = apply_opt_before_tuning(model, params, target)
    else:
        tvm_model = from_onnx(onnx_model)
    # Legalize any relax ops into tensorir.
    tvm_model = relax.transform.LegalizeOps()(tvm_model)
    if args.print:
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
    if args.print:
        print("Output...")
        print(tvm_res.numpy())


if __name__ == '__main__':
    main()
