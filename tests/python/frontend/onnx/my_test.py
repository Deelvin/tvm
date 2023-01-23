import scipy
import numpy as np

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib import graph_executor

import onnxruntime.backend
from onnx import TensorProto, helper, mapping


def get_input_data_shape_dict(graph_def, input_data):
    """Get input data shape"""
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            input_ = input_data[i]

            if input_ is None or not hasattr(input_, "shape") or input_.shape == ():
                # Skip adding input shape data when the input data is None;
                # This is to enable optional arguments for onnx operators.
                continue

            elif isinstance(input_, list):
                shape_dict[input_names[i]] = (len(input_),)

            else:
                shape_dict[input_names[i]] = input_.shape

    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def get_tvm_output_with_vm(
    graph_def,
    input_data,
    target,
    dev,
    opset=None,
    freeze_params=False,
    convert_config=None,
    validate_structural_equal=True,
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_onnx(
            graph_def,
            shape_dict,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_onnx(
                graph_def,
                shape_dict,
                opset=opset,
                freeze_params=freeze_params,
                convert_config=convert_config,
            )
        assert tvm.ir.structural_equal(mod, mod_with_span)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
        *input_data, **params
    )
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_tvm_output(
    graph_def,
    input_data,
    target,
    dev,
    output_shape=None,
    output_dtype="float32",
    opset=None,
    opt_level=1,
    convert_config=None,
):
    """Generic function to execute and get tvm output"""
    # TODO: Resolve the issues and remove the following lines
    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    print("GE: GET FROM ONNX")
    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, convert_config=convert_config
    )
    print("GE: MODULE:", mod)

    print("GE: BUILD LIB")
    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, params=params)

    print("GE: CREATE GE")
    m = graph_executor.create(graph, lib, dev)
    # set inputs
    print("GE: SET INPUTS")
    if isinstance(input_data, list):
        for i, _ in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            # pylint: disable=unnecessary-list-index-lookup
            m.set_input(input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    print("GE: SET PARAMS")
    m.set_input(**params)
    # execute
    print("GE: RUN")
    m.run()
    # get outputs
    print("GE: RETURN OUTPUTS")
    if isinstance(output_shape, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.numpy()


def get_onnxruntime_output(model, inputs):
    """Generic function to generate onnxruntime output"""
    rep = onnxruntime.backend.prepare(model.SerializeToString(), "CPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output


def verify_with_ort_with_inputs(
    model,
    inputs,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
    apply_softmax=False,
    opt_level=1,
    convert_config=None,
):
    """verify_with_ort_with_inputs"""
    if opset is not None:
        model.opset_import[0].version = opset

    ort_out = get_onnxruntime_output(model, inputs)
    if use_vm:
        tvm_out = get_tvm_output_with_vm(
            model,
            inputs,
            target,
            dev,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
    else:
        tvm_out = get_tvm_output(
            model,
            inputs,
            target,
            dev,
            out_shape,
            dtype,
            opset=opset,
            opt_level=opt_level,
            convert_config=convert_config,
        )

    if not isinstance(tvm_out, list):
        tvm_out = [tvm_out]
    if not isinstance(ort_out, list):
        ort_out = [ort_out]
    for tvm_val, ort_val in zip(tvm_out, ort_out):
        if apply_softmax:
            ort_val = scipy.special.softmax(ort_val)
            tvm_val = scipy.special.softmax(tvm_val)
        tvm.testing.assert_allclose(ort_val, tvm_val, rtol=rtol, atol=atol)
        assert ort_val.dtype == tvm_val.dtype


def test_maxunpool(target, dev):
    """test_maxunpool"""

    def verify_maxunpool(data, indices, kernel_shape, strides, output_shape=None, pads=None):
        input_names = ["xT", "xI"]
        input_info = [
            helper.make_tensor_value_info("xT", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("xI", TensorProto.INT64, list(indices.shape)),
        ]
        input_values = [data, indices]
        if output_shape is not None:
            input_names.append("output_shape")
            input_info.append(
                helper.make_tensor_value_info(
                    "output_shape", TensorProto.INT64, list(output_shape.shape)
                )
            )
            input_values.append(output_shape)
        else:
            # Compute expected output shape
            output_shape = np.asarray(([1, 1] + list(strides))) * np.asarray(list(data.shape))
            output_shape += np.asarray(([0, 0] + list(kernel_shape))) - np.asarray(
                ([0, 0] + list(strides))
            )
            if pads is not None:
                output_shape -= np.asarray(
                    [0, 0] + list(np.sum(np.reshape(list(pads), [-1, 2]), axis=-1))
                )
        output_shape = [int(i) for i in output_shape]

        node = helper.make_node(
            "MaxUnpool", inputs=input_names, outputs=["y"], kernel_shape=kernel_shape
        )

        if pads is not None:
            pad_attr = helper.make_attribute("pads", pads)
            node.attribute.append(pad_attr)

        if strides is not None:
            strides_attr = helper.make_attribute("strides", strides)
            node.attribute.append(strides_attr)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="size_test")

        verify_with_ort_with_inputs(
            model, input_values, use_vm=True, opset=11, target=target, dev=dev
        )

    # Basic test
    print("basic test")
    x_t = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
    x_i = np.array([[[[0, 7], [13, 15]]]], dtype=np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2])
    # Small stride
    print("small stride test")
    #verify_maxunpool(x_t, x_i, [2, 2], strides=[1, 1])
    # Big kernel
    print("big kernel test")
    verify_maxunpool(x_t, x_i, [3, 3], strides=[2, 2])
    # With output shape
    print("output shape test")
    output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
    #verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], output_shape=output_shape)
    # With explicit reverse padding
    print("explicit reverse padding test")
    pads = np.asarray([1, 1, 1, 1]).astype(np.int64)
    verify_maxunpool(x_t, x_i, [2, 2], strides=[2, 2], pads=pads)


def test_random_bernoulli(target, dev):
    """test_random_bernoulli"""

    def verify_bernoulli_with_ort(
        shape,
        in_dtype="float32",
        out_dtype="int32",
        seed=None,
        out_shape=None,
        target=target,
        dev=dev,
        use_vm=False,
        opset=None,
        freeze_params=False,
        rtol=0.1,
        atol=0.1,
        opt_level=1,
        convert_config=None,
    ):
        def get_bernoulli_model(shape, in_dtype="float32", out_dtype="int32", seed=None):
            onnx_itype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(in_dtype)]
            onnx_otype = mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_dtype)]
            node = helper.make_node(
                "Bernoulli",
                ["input"],
                ["output"],
            )
            dtype_attr = helper.make_attribute("dtype", onnx_otype)
            node.attribute.append(dtype_attr)
            if seed is not None:
                seed_attr = helper.make_attribute("seed", seed)
                node.attribute.append(seed_attr)

            graph = helper.make_graph(
                [node],
                "random_bernoulli_test",
                inputs=[helper.make_tensor_value_info("input", onnx_itype, list(shape))],
                outputs=[helper.make_tensor_value_info("output", onnx_otype, list(shape))],
            )
            return helper.make_model(graph, producer_name="random_bernoulli_test")

        inputs = np.random.uniform(size=shape).astype(in_dtype)
        print("GET BERNOULLI MODEL")
        model = get_bernoulli_model(shape, in_dtype, out_dtype, seed)
        if opset is not None:
            model.opset_import[0].version = opset

        print("GET ORT OUTPUT:")
        ort_out = get_onnxruntime_output(model, inputs)
        print(ort_out)
        if use_vm:
            print("GET VM OUTPUT:")
            tvm_out = get_tvm_output_with_vm(
                model,
                inputs,
                target,
                dev,
                opset=opset,
                freeze_params=freeze_params,
                convert_config=convert_config,
            )
        else:
            print("GET GE OUTPUT:")
            tvm_out = get_tvm_output(
                model,
                inputs,
                target,
                dev,
                out_shape,
                opset=opset,
                opt_level=opt_level,
                convert_config=convert_config,
            )
        print(tvm_out)

        if not isinstance(tvm_out, list):
            tvm_out = [tvm_out]
        if not isinstance(ort_out, list):
            ort_out = [ort_out]
        for tvm_val, ort_val in zip(tvm_out, ort_out):
            tvm.testing.assert_allclose(ort_val.mean(), tvm_val.mean(), rtol=rtol, atol=atol)
            tvm.testing.assert_allclose(np.std(ort_val), np.std(tvm_val), rtol=rtol, atol=atol)
            assert ort_val.dtype == tvm_val.dtype

    # Simple test
    print("Simple")
    verify_bernoulli_with_ort([100])

    # Floating output type
    print("Floating output")
    verify_bernoulli_with_ort([100], out_dtype="float32")

    # Double input type
    print("Double input")
    verify_bernoulli_with_ort([100], in_dtype="float64")

    # Test N-D tensor generation
    print("N-D tensor input")
    verify_bernoulli_with_ort([2, 4, 100, 100])

    # Test with seed
    print("Seed")
    verify_bernoulli_with_ort([100], seed=np.random.randint(1e6))


if __name__ == "__main__":
    target = "llvm -mcpu=core-avx2"
    dev = tvm.device(str(target), 0)
    test_random_bernoulli(target, dev)
    #test_maxunpool(target, dev)