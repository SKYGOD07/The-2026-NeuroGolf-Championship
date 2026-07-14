import os
import sys
import onnx
from onnx import helper, TensorProto
import numpy as np

# Add the root directory to path so we can import neurogolf_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import neurogolf_utils.neurogolf_utils as utils

def build_model():
    # 1. Define inputs & outputs
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    # 2. Define initializers (constants)
    slice_starts = helper.make_tensor("slice_starts", TensorProto.INT64, [2], [0, 0])
    slice_ends = helper.make_tensor("slice_ends", TensorProto.INT64, [2], [3, 3])
    slice_axes = helper.make_tensor("slice_axes", TensorProto.INT64, [2], [2, 3])

    slice_chan_starts = helper.make_tensor("slice_chan_starts", TensorProto.INT64, [1], [1])
    slice_chan_ends = helper.make_tensor("slice_chan_ends", TensorProto.INT64, [1], [10])
    slice_chan_axes = helper.make_tensor("slice_chan_axes", TensorProto.INT64, [1], [1])

    shape_A = helper.make_tensor("shape_A", TensorProto.INT64, [6], [1, 9, 3, 1, 3, 1])
    shape_B = helper.make_tensor("shape_B", TensorProto.INT64, [6], [1, 1, 1, 3, 1, 3])
    shape_out = helper.make_tensor("shape_out", TensorProto.INT64, [4], [1, 9, 9, 9])

    one_val = helper.make_tensor("one_val", TensorProto.FLOAT, [1], [1.0])

    pads_val = helper.make_tensor("pads_val", TensorProto.INT64, [8], [0, 0, 0, 0, 0, 0, 21, 21])

    # 3. Define nodes
    nodes = [
        # Slice input to X_sliced (1x10x3x3)
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["X_sliced"]),
        
        # Slice X_sliced to X_colors_sliced (1x9x3x3)
        helper.make_node("Slice", ["X_sliced", "slice_chan_starts", "slice_chan_ends", "slice_chan_axes"], ["X_colors_sliced"]),
        
        # Sum over channels of X_colors_sliced to get S (1x1x3x3)
        helper.make_node("ReduceSum", ["X_colors_sliced"], ["S"], axes=[1], keepdims=1),
        
        # Reshape X_colors_sliced to A_reshaped (1x9x3x1x3x1)
        helper.make_node("Reshape", ["X_colors_sliced", "shape_A"], ["A_reshaped"]),
        
        # Reshape S to B_reshaped (1x1x1x3x1x3)
        helper.make_node("Reshape", ["S", "shape_B"], ["B_reshaped"]),
        
        # Multiply A_reshaped and B_reshaped to get Y_colors_6d (1x9x3x3x3x3)
        helper.make_node("Mul", ["A_reshaped", "B_reshaped"], ["Y_colors_6d"]),
        
        # Reshape Y_colors_6d to Y_colors (1x9x9x9)
        helper.make_node("Reshape", ["Y_colors_6d", "shape_out"], ["Y_colors"]),
        
        # ReduceSum Y_colors over channels to get S_out (1x1x9x9)
        helper.make_node("ReduceSum", ["Y_colors"], ["S_out"], axes=[1], keepdims=1),
        
        # Subtract S_out from 1.0 to get Y_chan0 (1x1x9x9)
        helper.make_node("Sub", ["one_val", "S_out"], ["Y_chan0"]),
        
        # Concat Y_chan0 and Y_colors along axis 1 to get Y_final_9x9 (1x10x9x9)
        helper.make_node("Concat", ["Y_chan0", "Y_colors"], ["Y_final_9x9"], axis=1),
        
        # Pad Y_final_9x9 to output (1x10x30x30)
        helper.make_node("Pad", ["Y_final_9x9"], ["output"], pads=[0, 0, 0, 0, 0, 0, 21, 21], value=0.0),
    ]

    # 4. Make Graph and Model
    graph = helper.make_graph(
        nodes,
        "kronecker_graph",
        [x],
        [y],
        initializer=[
            slice_starts, slice_ends, slice_axes,
            slice_chan_starts, slice_chan_ends, slice_chan_axes,
            shape_A, shape_B, shape_out,
            one_val
        ]
    )

    model = helper.make_model(graph, ir_version=utils._IR_VERSION, opset_imports=utils._OPSET_IMPORTS)
    return model

if __name__ == "__main__":
    task_num = 1
    model = build_model()
    # Set correct neurogolf dir for local execution
    utils._NEUROGOLF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/"
    examples = utils.load_examples(task_num)
    utils.verify_network(model, task_num, examples)

