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

    # 2. Define initializers
    # slice input to channel 1, 6 rows, 3 cols
    slice_starts = helper.make_tensor("slice_starts", TensorProto.INT64, [3], [1, 0, 0])
    slice_ends = helper.make_tensor("slice_ends", TensorProto.INT64, [3], [2, 6, 3])
    slice_axes = helper.make_tensor("slice_axes", TensorProto.INT64, [3], [1, 2, 3])

    # row slice indices (axis 2)
    start0 = helper.make_tensor("start0", TensorProto.INT64, [1], [0])
    end0 = helper.make_tensor("end0", TensorProto.INT64, [1], [1])
    start1 = helper.make_tensor("start1", TensorProto.INT64, [1], [1])
    end1 = helper.make_tensor("end1", TensorProto.INT64, [1], [2])
    start2 = helper.make_tensor("start2", TensorProto.INT64, [1], [2])
    end2 = helper.make_tensor("end2", TensorProto.INT64, [1], [3])
    start3 = helper.make_tensor("start3", TensorProto.INT64, [1], [3])
    end3 = helper.make_tensor("end3", TensorProto.INT64, [1], [4])
    axis_row = helper.make_tensor("axis_row", TensorProto.INT64, [1], [2])

    # constants
    one_const = helper.make_tensor("one_const", TensorProto.FLOAT, [1], [1.0])
    zero_const = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])

    # ones_9x3 (active grid mask of shape 1x1x9x3)
    ones_9x3 = helper.make_tensor("ones_9x3", TensorProto.FLOAT, [1, 1, 9, 3], [1.0] * 27)

    # 3. Define nodes
    nodes = [
        # Slice input channel 1 (1x1x6x3)
        helper.make_node("Slice", ["input", "slice_starts", "slice_ends", "slice_axes"], ["X_c1"]),

        # Slice individual rows from X_c1 (each is 1x1x1x3)
        helper.make_node("Slice", ["X_c1", "start0", "end0", "axis_row"], ["Row0"]),
        helper.make_node("Slice", ["X_c1", "start1", "end1", "axis_row"], ["Row1"]),
        helper.make_node("Slice", ["X_c1", "start2", "end2", "axis_row"], ["Row2"]),
        helper.make_node("Slice", ["X_c1", "start3", "end3", "axis_row"], ["Row3"]),

        # Compute D = sum(abs(Row0 - Row3))
        helper.make_node("Sub", ["Row0", "Row3"], ["Diff"]),
        helper.make_node("Abs", ["Diff"], ["Abs_Diff"]),
        helper.make_node("ReduceSum", ["Abs_Diff"], ["D"], axes=[3], keepdims=1),

        # Compute P = 1.0 - clip(D, 0.0, 1.0)
        helper.make_node("Clip", ["D"], ["D_clipped"], min=0.0, max=1.0),
        helper.make_node("Sub", ["one_const", "D_clipped"], ["P"]),

        # Compute 1.0 - P
        helper.make_node("Sub", ["one_const", "P"], ["one_minus_P"]),

        # Compute Row6 = P * Row0 + (1.0 - P) * Row2
        helper.make_node("Mul", ["P", "Row0"], ["P_Row0"]),
        helper.make_node("Mul", ["one_minus_P", "Row2"], ["one_minus_P_Row2"]),
        helper.make_node("Add", ["P_Row0", "one_minus_P_Row2"], ["Row6"]),

        # Compute Row7 = P * Row1 + (1.0 - P) * Row3
        helper.make_node("Mul", ["P", "Row1"], ["P_Row1"]),
        helper.make_node("Mul", ["one_minus_P", "Row3"], ["one_minus_P_Row3"]),
        helper.make_node("Add", ["P_Row1", "one_minus_P_Row3"], ["Row7"]),

        # Compute Row8 = P * Row2 + (1.0 - P) * Row0
        helper.make_node("Mul", ["P", "Row2"], ["P_Row2"]),
        helper.make_node("Mul", ["one_minus_P", "Row0"], ["one_minus_P_Row0"]),
        helper.make_node("Add", ["P_Row2", "one_minus_P_Row0"], ["Row8"]),

        # Concat original 6 rows and Row6, Row7, Row8 along axis 2 (height) to get B_2_9x3 (1x1x9x3)
        helper.make_node("Concat", ["X_c1", "Row6", "Row7", "Row8"], ["B_2_9x3"], axis=2),

        # Pad B_2_9x3 to 30x30 with 0.0 -> B_2
        helper.make_node("Pad", ["B_2_9x3"], ["B_2"], pads=[0, 0, 0, 0, 0, 0, 21, 27], value=0.0),

        # Pad ones_9x3 to 30x30 with 0.0 -> G_out (active grid mask)
        helper.make_node("Pad", ["ones_9x3"], ["G_out"], pads=[0, 0, 0, 0, 0, 0, 21, 27], value=0.0),

        # B_0 = G_out * (1.0 - B_2)
        helper.make_node("Sub", ["one_const", "B_2"], ["one_minus_B_2"]),
        helper.make_node("Mul", ["G_out", "one_minus_B_2"], ["B_0"]),

        # zeros_30 = G_out * 0.0
        helper.make_node("Mul", ["G_out", "zero_const"], ["zeros_30"]),

        # Concat channels: 0: B_0, 1: zeros, 2: B_2, 3-9: zeros
        helper.make_node("Concat", [
            "B_0", "zeros_30", "B_2",
            "zeros_30", "zeros_30", "zeros_30", "zeros_30", "zeros_30", "zeros_30", "zeros_30"
        ], ["output"], axis=1)
    ]

    # 4. Make Graph and Model
    graph = helper.make_graph(
        nodes,
        "pattern_graph",
        [x],
        [y],
        initializer=[
            slice_starts, slice_ends, slice_axes,
            start0, end0, start1, end1, start2, end2, start3, end3, axis_row,
            one_const, zero_const, ones_9x3
        ]
    )

    model = helper.make_model(graph, ir_version=utils._IR_VERSION, opset_imports=utils._OPSET_IMPORTS)
    return model

if __name__ == "__main__":
    task_num = 3
    model = build_model()
    # Set correct neurogolf dir for local execution
    utils._NEUROGOLF_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tasks_data")) + "/"
    examples = utils.load_examples(task_num)
    utils.verify_network(model, task_num, examples)
    
    # Move the generated onnx file to the onnx_models folder
    src = f"task{task_num:03d}.onnx"
    dst = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "onnx_models", src))
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print(f"Moved {src} to onnx_models/")
