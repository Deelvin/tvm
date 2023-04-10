import torch
import torch_mlir
import iree_torch

from bert_utils import Profiler, get_encoded_input, get_pytorch_model, NUM_ITERATIONS


def main():
    model = get_pytorch_model()
    encoded_input = get_encoded_input(return_tensors='pt')

    # PyTorch
    with Profiler("PyTorch", show_latency=True, iterations_number=NUM_ITERATIONS):
        for _ in range(NUM_ITERATIONS):
            output = model(**encoded_input)
    # print(output)

    # IREE
    encoded_input = [torch.tensor(inp) for inp in encoded_input.values()]

    print("Compiling with Torch-MLIR")
    linalg_on_tensors_mlir = torch_mlir.compile(
        model,
        encoded_input,
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS,
        use_tracing=True,
        verbose=False,
    )

    print("Compiling with IREE")
    iree_backend = "llvm-cpu"
    iree_vmfb = iree_torch.compile_to_vmfb(linalg_on_tensors_mlir, iree_backend)

    print("Loading in IREE")
    invoker = iree_torch.load_vmfb(iree_vmfb, iree_backend)

    print("Running on IREE")
    for _ in range(5):
        invoker.forward(*encoded_input)
    with Profiler("IREE", show_latency=True, iterations_number=NUM_ITERATIONS):
        for _ in range(NUM_ITERATIONS):
            result = invoker.forward(*encoded_input)
    # print(result)


if __name__ == '__main__':
    main()
