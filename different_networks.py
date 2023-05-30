import os
import tvm
from tvm import relay
from tvm import relay, auto_scheduler, autotvm, transform
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm import meta_schedule as ms
from tvm import rpc

from tvm.relay.backend import Executor

from tvm.contrib import utils, ndk
import onnx
from tvm.contrib import graph_executor

import argparse
import yaml

class MyDumper(yaml.Dumper):
    
    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)


def add_common_args(parser):
    common_args = parser.add_argument_group('Common arguments')
    common_args.add_argument('--input_model', type=str, choices=[
        'resnet_obfuscated',
        'resnet_quant_obfuscated',
        'srgan_obfuscated',
        'srgan_quant_obfuscated',
        'inception_obfuscated',
        'inception_quant_obfuscated',
        'deeplab_obfuscated',
        'deeplab_qat_quant_obfuscated',
        'inception_v1_quant_tflite',
        'inception_v3_tflite',
        'inception_v3_quant_tflite',
        'mobilenet_v1_tflite',
        'mobilenet_v1_quant_tflite',
        'mobilenet_v2_tflite',
        'mobilenet_v2_quant_tflite',
        'all'
        ], default='resnet_obfuscated')
    common_args.add_argument('--model_dir', type=str, default='')
    common_args.add_argument('--target', type=str, default="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod")
    common_args.add_argument('--target_host', type=str, default='llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod')
    # common_args.add_argument('--precision', type=str, choices=['float16','float16_acc32','float32'], default='float32')
    common_args.add_argument('--strategy_name', type=str, choices=['replay-trace', 'replay-func', 'evolutionary'], default='replay-trace')
    common_args.add_argument('--module_equality_name', type=str, choices=['ignore-ndarray', 'structural', 'anchor-block'], default='ignore-ndarray')
    common_args.add_argument('--work_dir', type=str)
    common_args.add_argument('--tune', type=bool, default=False)
    common_args.add_argument('--trials_global', type=int, default=7936)
    common_args.add_argument('--trials_per_task', type=int, default=256)
    common_args.add_argument('--trials_per_iter', type=int, default=64)
    common_args.add_argument('--key', type=str, default="android")
    common_args.add_argument('--host', type=str, default="0.0.0.0")
    common_args.add_argument('--port', type=int, default=9190)


def build_arguments_parser():
    parser = argparse.ArgumentParser(description='ARM CPU Compile&Tune', allow_abbrev=False)
    add_common_args(parser)

    return parser

def run_tuning(tasks, task_weights, log_file, num_measure_trials = 20000):
    print("Begin tuning...")
    measure_runner = auto_scheduler.RPCRunner(
        args.key,
        args.host,
        args.port,
        min_repeat_ms=300,
        timeout=30,
        repeat=2
    )
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    builder=auto_scheduler.LocalBuilder(build_func=ndk.create_shared, timeout=15)
    tune_option = auto_scheduler.TuningOptions(
        builder=builder,
        num_measure_trials=num_measure_trials,
        num_measures_per_round = 32, # to speed-up round-robin measurements
        runner=measure_runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        verbose=2
    )
    tuner.tune(tune_option)

def tune_ansor(mod, params, name):  
    print("Tune ansor:", name) 
    log_file = name + ".ansor.json"

    if not os.path.exists(log_file):
        tasks, task_weights = auto_scheduler.extract_tasks(
            mod["main"], params, target=args.target, target_host=args.target_host)
        for idx, task in enumerate(tasks):
            print("========== Task %d  (workload key: %s) ==========" %
                  (idx, task.workload_key))
            print(task.compute_dag)

        run_tuning(tasks, task_weights, log_file, num_measure_trials = 512)

def compile_ansor(mod, params, name):
    print("Compile ansor:", name)
    log_file = name + ".ansor.json"
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=args.target,
                                target_host=args.target_host, params=params)
            
    export_path = os.path.join(args.model_dir, name + ".ansor.so")
    lib.export_library(export_path, ndk.create_shared)


def compile_default(mod, params, name):
    print("Compile default:", name)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=args.target, target_host=args.target_host, params=params)

    export_path = os.path.join(args.model_dir, name + ".so")
    lib.export_library(export_path, ndk.create_shared)


def tune_kernels_autotvm(
    tasks, measure_option, tuner="xgb", n_trial=333, log_filename="tuning.log"
):

    for i, task in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb" or tuner == "xgb-rank":
            tuner_obj = XGBTuner(task, loss_type="rank")
        elif tuner == "ga":
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(task)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial = min(n_trial, len(task.config_space))
        print ("Number of trials as len of config_space: " + str(n_trial))
        
        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=None,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix=prefix),
                autotvm.callback.log_to_file(log_filename),
            ],
        )

def tune_autotvm(mod, params, name):
    print("Tune autotvm:", name)
    # extract workloads from relay program
    print("Extract autotvm tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"], target=args.target_host, target_host=args.target_host, params=params)
    for idx, task in enumerate(tasks):
        print("========== Task %d ==========" %
                (idx))
        print(task)

    # run tuning tasks
    atvmMeasureOptions = autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func=ndk.create_shared, timeout=15),
        # runner=autotvm.LocalRunner(
        #     number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        # ))
        runner=autotvm.RPCRunner(
            args.key,
            host=args.host,
            port=args.port,
            number=50,
            timeout=15,
            #min_repeat_ms=150,
            #cooldown_interval=150
        ))

    log_file = name + ".atvm.json"
    tune_kernels_autotvm(tasks,
        log_filename = log_file,
        tuner = "xgb",
        n_trial = 32,
        measure_option = atvmMeasureOptions)


def compile_autotvm(mod, params, name):
    log_file = name + ".atvm.json"
    # compile kernels with graph-level best records
    with autotvm.apply_history_best(log_file):
        print("Compile autotvm:", name)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build_module.build(mod, target=args.target, target_host = args.target_host, params=params)
            
    export_path = os.path.join(args.model_dir, name + ".atvm.so")
    lib.export_library(export_path, ndk.create_shared)

    #     # upload parameters to device
    #     dev = tvm.cpu()
    #     data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
    #     module = runtime.GraphModule(lib["default"](dev))
    #     module.set_input(input_name, data_tvm)

    #     # evaluate
    #     print("Evaluate inference time cost...")
    #     ftimer = module.module.time_evaluator("run", dev, number=100, repeat=3)
    #     prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
    #     print(
    #         "Mean inference time (std dev): %.2f ms (%.2f ms)"
    #         % (np.mean(prof_res), np.std(prof_res))
    #     )

def tune_metaschedule(mod, params, name):
    print("Tune metaschedule:", name)
    database = None
    
    work_dir = args.work_dir
    if not work_dir:
        work_dir = f"./database_logs_{args.strategy_name}/{name}/"
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    args.work_dir = work_dir
    
    target_llvm = tvm.target.Target(args.target + ' -num-cores 4', host=args.target_host)

    executor = Executor("graph", {"link-params": True})
    # This line is necessary for link-params to take effect during
    # task extraction and relay.build(...).
    mod = mod.with_attr("executor", executor)

    ms_builder = ms.builder.LocalBuilder(f_export=ndk.create_shared, timeout_sec=60)

    rpc_config=ms.runner.RPCConfig(
                    tracker_host=args.host,
                    tracker_port=args.port,
                    tracker_key=args.key,
                    session_timeout_sec=6000,
                )

    evaluator_config=ms.runner.EvaluatorConfig(
        number=3,
        repeat=1,
        min_repeat_ms=100,
        enable_cpu_cache_flush=False,
    )

    ms_rpc_runner = ms.runner.RPCRunner(rpc_config=rpc_config,
                evaluator_config=evaluator_config,
                alloc_repeat=1,
            )
    
    if not args.tune:
        args.trials_global = args.trials_per_task = args.trials_per_iter = 0
    
    database = ms.relay_integration.tune_relay(
        mod=mod,
        target=target_llvm,
        params=params,
        work_dir=work_dir,
        max_trials_global=args.trials_global,
        max_trials_per_task=args.trials_per_task,
        num_trials_per_iter=args.trials_per_iter,
        strategy=args.strategy_name,
        seed=1,
        # builder=builder,
        runner=ms_rpc_runner,
        module_equality=args.module_equality_name,
    )
    
    return executor, database, target_llvm
    
def compile_metaschedule(mod, params, name, executor, database, target):
    print("Compile metaschedule:", name)
    if not database:
        print("Load database from disk...")
        database = ms.database.JSONDatabase(f"{args.work_dir}/database_workload.json", f"{args.work_dir}/database_tuning_record.json", module_equality=args.module_equality_name, allow_missing=False)
    
    cpu_tuned = ms.relay_integration.compile_relay(
        database=database,
        mod=mod,
        target=target,
        params=params,
        executor=executor,
    )
    
    from tvm import te, tir
    from tvm.script import tir as T

    from tvm.tir.tensor_intrin.arm_cpu import ARM_DOT_4x4_i8_SDOT_INTRIN
    
    workload = database.commit_workload(mod)
    print("BDS commit_workload end", flush=True)
    sch = tir.Schedule(mod, debug_mask="all") 
    records = database.get_top_k(workload,10)
    
    for record in records:
        if record.run_secs:
            print("BDS rec run_secs", [v.value for v in record.run_secs], flush=True)
        else:
            print("BDS rec run_secs", [])
        print("BDS Record Trace", record.trace)
        new_sch = sch.copy()
        try:
            record.trace.apply_to_schedule(new_sch, remove_postproc=True)
            print("BDS Schedule with applied trace", new_sch.mod)
        except:  
            print("BDS Failed to apply trace to schedule!")   
            pass
        
    
    try:
        database.get_top_k(workload,1)[0].trace.apply_to_schedule(sch, remove_postproc=True)
    except:
        pass
        
    print("BDS default schedule", sch.mod, flush=True)
    #print("BDS top schedule", new_sch.mod, flush=True)
    print("BDS get_top_k", database.get_top_k(workload,1), flush=True)
    
    export_path = os.path.join(args.model_dir, name + ".ms.so")
    cpu_tuned.export_library(export_path, ndk.create_shared)
    print(cpu_tuned.get_lib().get_source("asm"))

def benchmark(mod, extension):
    if len(extension):
        input_path = mod + "." + extension + ".so"
    else:
        input_path = mod + ".so"
        
    if args.model_dir:
        input_path = os.path.join(args.model_dir, input_path)
    print("measuring " + input_path)

    tracker = rpc.connect_tracker(args.host, args.port)
    remote = tracker.request(
            args.key , priority=0, session_timeout=None
        )
    ctx = remote.cpu(0)
    remote.upload(input_path)
    lib = remote.load_module(os.path.basename(input_path))
    m = graph_executor.GraphModule(lib["default"](ctx))
    ftimer = m.module.time_evaluator("run", ctx, repeat=10, min_repeat_ms=500)
    prof_res = np.array(ftimer().results) * 1e3  # convert to millisecond
    print(
        "Mean inference time (std dev): %.2f ms (%.2f ms)" % (np.mean(prof_res), np.std(prof_res))
    )
        

def compile(mod, params, name):
    #compile_default(mod, params, name)
    # tune_ansor(mod, params, name)
    # compile_ansor(mod, params, name)
    # tune_autotvm(mod, params, name)
    # compile_autotvm(mod, params, name)
    executor, database, target = tune_metaschedule(mod, params, name)
    compile_metaschedule(mod, params, name, executor, database, target)
    benchmark(name, "")
    # benchmark(name, "ansor")
    # benchmark(name, "atvm")
    benchmark(name, "ms")


def resnet_quant_obfuscated():
    name = "resnet_quantized_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["data"] = [1,3,224,224]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)


def resnet_obfuscated():
    name = "resnet_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["data"] = [1,3,224,224]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)


def inception_quant_obfuscated():
    name = "inception_quantized_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["Mul:0"] = [1, 299, 299, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)


def inception_obfuscated():
    name = "inception_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["Mul:0"] = [1, 299, 299, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)

def deeplab_qat_quant_obfuscated():
    name = "deeplab_qat_quantized_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["ImageTensor:0"] = [1, 512, 512, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)


def deeplab_obfuscated():
    name = "deeplab_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["ImageTensor:0"] = [1, 512, 512, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)

def srgan_quant_obfuscated():
    name = "srgan_quantized_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["input"] = [1, 128,128, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)

def srgan_obfuscated():
    name = "srgan_obfuscated"
    path = os.path.join((args.model_dir), name+".onnx")
    onnx_model = onnx.load(path)
    shape_dict = {}
    shape_dict["input"] = [1, 128,128, 3]
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict, freeze_params=True)

    compile(mod, params, name)
    

### TFLITE MODELS ###

def parse_tflite_model(tflite_model_file, input_name, input_shape, dtype="float32"):
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        tflite_model, shape_dict={input_name: input_shape}, dtype_dict={input_name: dtype}
    )

    return mod, params


def inception_v1_quant_tflite():
    name = "inception_v1_224_quant"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "uint8"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)

def inception_v3_tflite():
    name = "inception_v3"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 299, 299, 3)
    input_dtype = "float32"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)

def inception_v3_quant_tflite():
    name = "inception_v3_quant"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 299, 299, 3)
    input_dtype = "uint8"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name) 

def mobilenet_v1_tflite():
    name = "mobilenet_v1_1.0_224"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "float32"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)
    

def mobilenet_v1_quant_tflite():
    name = "mobilenet_v1_1.0_224_quant"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "uint8"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)

def mobilenet_v2_tflite():
    name = "mobilenet_v2_1.0_224"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "float32"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)

def mobilenet_v2_quant_tflite():
    name = "mobilenet_v2_1.0_224_quant"
    path = os.path.join((args.model_dir), name+".tflite")
        
    input_name = "input"
    input_shape = (1, 224, 224, 3)
    input_dtype = "uint8"
    
    mod, params = parse_tflite_model(path, input_name, input_shape, input_dtype)


    compile(mod, params, name)
    



if __name__ == "__main__":
    # parse arguments
    parser = build_arguments_parser()
    args = parser.parse_args()
    
    if not args.input_model:
        print("Please specify the model or set --input-model=all to run all available models.")
        exit(1)        
    elif args.input_model == 'all':
        # resnet_obfuscated()
        # resnet_quant_obfuscated()
        # inception_obfuscated()
        # inception_quant_obfuscated()
        # deeplab_obfuscated()
        # deeplab_qat_quant_obfuscated()
        # srgan_obfuscated()
        # srgan_quant_obfuscated()
        # inception_v1_quant_tflite()
        # inception_v3_tflite()
        # inception_v3_quant_tflite()
        # mobilenet_v1_tflite()
        # args.work_dir = "./database_logs_default_llvm_arm"
        # args.target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
        # args.target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"
        # print("Run 1: default LLVM")
        # mobilenet_v1_quant_tflite()
        # args.work_dir = "./database_logs_+neon+v8.2a_arm"
        # args.target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a"
        # args.target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a"
        # print("Run 2: +neon")
        # mobilenet_v1_quant_tflite()
        # args.work_dir = "./database_logs_+dotprod+v8.2a_arm"
        # args.target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        # args.target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        # print("Run 3: +dotprod")
        # mobilenet_v1_quant_tflite()
        args.work_dir = "./database_logs_sdot"
        args.target = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod"
        args.target_host = "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod"
        print("Run 4: +neon +dotprod")
        mobilenet_v1_quant_tflite()
        # mobilenet_v2_tflite()
        # mobilenet_v2_quant_tflite()
    else:
        locals()[args.input_model]()