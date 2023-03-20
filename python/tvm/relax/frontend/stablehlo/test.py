import numpy as np
import tvm
import tvm.testing

from tvm import relax
from tvm.relax.testing import nn


def get_exec(data_shape):
    builder = relax.BlockBuilder()
    weight1_np = np.random.randn(64, 64).astype("float32")
    weight2_np = np.random.randn(64, 64).astype("float32")

    with builder.function("main"):
        model = nn.Sequential(
            nn.Linear(data_shape[1], weight1_np.shape[0], bias=False),
            nn.ReLU(),
            nn.Linear(weight2_np.shape[0], weight2_np.shape[1], bias=False),
            nn.ReLU(),
        )
        data = nn.Placeholder(data_shape, name="data")
        output = model(data)
        params = [data] + model.parameters()
        builder.emit_func_output(output, params=params)

    mod = builder.get()
    mod.show()


get_exec
