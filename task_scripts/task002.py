import os
import sys
import onnx
from onnx import helper, TensorProto
import numpy as np

# Add the root directory to path so we can import neurogolf_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import neurogolf_utils.neurogolf_utils as utils

def build_model(num_steps=30):
    # 1. Define inputs & outputs
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    # 2. Define initializers
    # slice constants
    slice_starts = helper.make_tensor("slice_starts", TensorProto.INT64, [3], [0, 1, 1])
    slice_ends = helper.make_tensor("slice_ends", TensorProto.INT64, [3], [1, 29, 29])
    slice_axes = helper.make_tensor("slice_axes", TensorProto.INT64, [3], [1, 2, 3])

    slice_chan3_starts = helper.make_tensor("slice_chan3_starts", TensorProto.INT64, [1], [3])
    slice_chan3_ends = helper.make_tensor("slice_chan3_ends", TensorProto.INT64, [1], [4])
    slice_chan3_axes = helper.make_tensor("slice_chan3_axes", TensorProto.INT64, [1], [1])

    # kernel for 4-connectivity dilation (up, down, left, right, center)
    # shape [1, 1, 3, 3]
    kernel_weights = [0.0, 1.0, 0.0,
                      1.0, 1.0, 1.0,
                      0.0, 1.0, 0.0]
    W_conv = helper.make_tensor("W_conv", TensorProto.FLOAT, [1, 1, 3, 3], kernel_weights)

    # constants
    zero_const = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])
    one_const = helper.make_tensor("one_const", TensorProto.FLOAT, [1], [1.0])

    # 3. Define nodes
    nodes = [
        # Get active grid mask G = sum over all channels of input
        helper.make_node("ReduceSum", ["input"], ["G"], axes=[1], keepdims=1),

        # Get wall mask W = input[:, 3:4, :, :]
        helper.make_node("Slice", ["input", "slice_chan3_starts", "slice_chan3_ends", "slice_chan3_axes"], ["W"]),

        # Get one_minus_W = 1.0 - W
        helper.make_node("Sub", ["one_const", "W"], ["one_minus_W"]),

        # Create border mask:
        # a. Slice input channel 0 to 1x1x28x28
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["zeros_28"]),
        # b. Multiply by 0.0 to get 1x1x28x28 zeros
        helper.make_node("Mul", ["zeros_28", "zero_const"], ["zeros_28_scaled"]),
        # c. Pad with 1.0 to get 1x1x30x30 border mask
        helper.make_node("Pad", ["zeros_28_scaled"], ["border"], pads=[0, 0, 1, 1, 0, 0, 1, 1], value=1.0),

        # F_0 = border * (1.0 - W)
        helper.make_node("Mul", ["border", "one_minus_W"], ["F_0"]),
    ]

    # Propagation loop
    F_current = "F_0"
    for t in range(num_steps):
        conv_out = f"F_conv_{t}"
        clipped_out = f"F_clipped_{t}"
        next_out = f"F_next_{t}"
        nodes.append(helper.make_node("Conv", [F_current, "W_conv"], [conv_out], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("Clip", [conv_out], [clipped_out], min=0.0, max=1.0))
        nodes.append(helper.make_node("Mul", [clipped_out, "one_minus_W"], [next_out]))
        F_current = next_out

    # Final flooded mask F_final
    nodes.extend([
        # F_grid = F_final * G (flooded pixels inside active grid)
        helper.make_node("Mul", [F_current, "G"], ["F_grid"]),

        # B_0 = F_grid * (1.0 - W)
        helper.make_node("Mul", ["F_grid", "one_minus_W"], ["B_0"]),

        # B_3 = W
        # B_4 = G - F_grid - W
        helper.make_node("Add", ["F_grid", "W"], ["F_grid_plus_W"]),
        helper.make_node("Sub", ["G", "F_grid_plus_W"], ["B_4"]),

        # zeros_30 = G * 0.0
        helper.make_node("Mul", ["G", "zero_const"], ["zeros_30"]),

        # Concat output channels
        # channels: 0 (B_0), 1 (zeros), 2 (zeros), 3 (W), 4 (B_4), 5-9 (zeros)
        helper.make_node("Concat", [
            "B_0", "zeros_30", "zeros_30", "W", "B_4",
            "zeros_30", "zeros_30", "zeros_30", "zeros_30", "zeros_30"
        ], ["output"], axis=1)
    ])

    # 4. Make Graph and Model
    graph = helper.make_graph(
        nodes,
        "fill_graph",
        [x],
        [y],
        initializer=[
            slice_starts, slice_ends, slice_axes,
            slice_chan3_starts, slice_chan3_ends, slice_chan3_axes,
            W_conv, zero_const, one_const
        ]
    )

    model = helper.make_model(graph, ir_version=utils._IR_VERSION, opset_imports=utils._OPSET_IMPORTS)
    return model

if __name__ == "__main__":
    task_num = 2
    model = build_model(num_steps=25)
    # Set correct neurogolf dir for local execution
    utils._NEUROGOLF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/"
    examples = utils.load_examples(task_num)
    utils.verify_network(model, task_num, examples)
