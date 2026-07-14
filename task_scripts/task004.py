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
    # slice axes
    slice_chan_axes = helper.make_tensor("slice_chan_axes", TensorProto.INT64, [1], [1])
    slice_h_axes = helper.make_tensor("slice_h_axes", TensorProto.INT64, [1], [2])
    slice_w_axes = helper.make_tensor("slice_w_axes", TensorProto.INT64, [1], [3])

    # index slicing constants
    start_0 = helper.make_tensor("start_0", TensorProto.INT64, [1], [0])
    end_30 = helper.make_tensor("end_30", TensorProto.INT64, [1], [30])
    start_1 = helper.make_tensor("start_1", TensorProto.INT64, [1], [1])
    end_31 = helper.make_tensor("end_31", TensorProto.INT64, [1], [31])

    # constants
    one_const = helper.make_tensor("one_const", TensorProto.FLOAT, [1], [1.0])
    zero_const = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])

    # K_left 1D kernel: [0]*29 + [1]*30 (shape [1, 1, 1, 59])
    kernel_vals = [0.0] * 29 + [1.0] * 30
    K_left = helper.make_tensor("K_left", TensorProto.FLOAT, [1, 1, 1, 59], kernel_vals)

    # channel slicing starts/ends for each channel 1..9
    starts_ends = {}
    for c in range(1, 10):
        starts_ends[f"start_c{c}"] = helper.make_tensor(f"start_c{c}", TensorProto.INT64, [1], [c])
        starts_ends[f"end_c{c}"] = helper.make_tensor(f"end_c{c}", TensorProto.INT64, [1], [c+1])

    # 3. Define nodes
    nodes = []
    y_channels = []

    # Get active grid mask G_grid = sum over all 10 channels of input
    nodes.append(helper.make_node("ReduceSum", ["input"], ["G_grid"], axes=[1], keepdims=1))

    for c in range(1, 10):
        # a. Slice channel c to X_c
        nodes.append(helper.make_node("Slice", ["input", f"start_c{c}", f"end_c{c}", "slice_chan_axes"], [f"X_c{c}"]))

        # b. Get V_c = ReduceMax(X_c, axes=[3])
        nodes.append(helper.make_node("ReduceMax", [f"X_c{c}"], [f"V_c{c}"], axes=[3], keepdims=1))

        # c. Shift V_c up to get V_up_c
        nodes.append(helper.make_node("Pad", [f"V_c{c}"], [f"V_padded_c{c}"], pads=[0, 0, 0, 0, 0, 0, 1, 0], value=0.0))
        nodes.append(helper.make_node("Slice", [f"V_padded_c{c}", "start_1", "end_31", "slice_h_axes"], [f"V_up_c{c}"]))

        # d. M_bottom_row_c = V_c * (1.0 - V_up_c)
        nodes.append(helper.make_node("Sub", ["one_const", f"V_up_c{c}"], [f"one_minus_V_up_c{c}"]))
        nodes.append(helper.make_node("Mul", [f"V_c{c}", f"one_minus_V_up_c{c}"], [f"M_bottom_row_c{c}"]))

        # e. B_c = X_c * M_bottom_row_c
        nodes.append(helper.make_node("Mul", [f"X_c{c}", f"M_bottom_row_c{c}"], [f"B_c{c}"]))

        # f. Shift B_c left to get B_left_c
        nodes.append(helper.make_node("Pad", [f"B_c{c}"], [f"B_padded_c{c}"], pads=[0, 0, 0, 0, 0, 0, 0, 1], value=0.0))
        nodes.append(helper.make_node("Slice", [f"B_padded_c{c}", "start_1", "end_31", "slice_w_axes"], [f"B_left_c{c}"]))

        # g. M_rightmost_pixel_c = B_c * (1.0 - B_left_c)
        nodes.append(helper.make_node("Sub", ["one_const", f"B_left_c{c}"], [f"one_minus_B_left_c{c}"]))
        nodes.append(helper.make_node("Mul", [f"B_c{c}", f"one_minus_B_left_c{c}"], [f"M_rightmost_pixel_c{c}"]))

        # h. col_mask_c = ReduceMax(M_rightmost_pixel_c, axes=[2])
        nodes.append(helper.make_node("ReduceMax", [f"M_rightmost_pixel_c{c}"], [f"col_mask_c{c}"], axes=[2], keepdims=1))

        # i. M_left_col_c = Conv(col_mask_c, K_left)
        nodes.append(helper.make_node("Conv", [f"col_mask_c{c}", "K_left"], [f"M_left_col_c{c}"], pads=[0, 29, 0, 29]))

        # k. Shift X_c right to get X_shifted_c
        nodes.append(helper.make_node("Pad", [f"X_c{c}"], [f"X_padded_c{c}"], pads=[0, 0, 0, 1, 0, 0, 0, 0], value=0.0))
        nodes.append(helper.make_node("Slice", [f"X_padded_c{c}", "start_0", "end_30", "slice_w_axes"], [f"X_shifted_c{c}"]))

        # l. Y_other_c = Max(X_c * col_mask_c, X_shifted_c * M_left_col_c)
        nodes.append(helper.make_node("Mul", [f"X_c{c}", f"col_mask_c{c}"], [f"X_c_R{c}"]))
        nodes.append(helper.make_node("Mul", [f"X_shifted_c{c}", f"M_left_col_c{c}"], [f"X_shifted_L{c}"]))
        nodes.append(helper.make_node("Max", [f"X_c_R{c}", f"X_shifted_L{c}"], [f"Y_other_c{c}"]))

        # m. Y_c = X_c * M_bottom_row_c + Y_other_c * (1.0 - M_bottom_row_c)
        nodes.append(helper.make_node("Sub", ["one_const", f"M_bottom_row_c{c}"], [f"one_minus_M_bottom_c{c}"]))
        nodes.append(helper.make_node("Mul", [f"X_c{c}", f"M_bottom_row_c{c}"], [f"Y_bottom_c{c}"]))
        nodes.append(helper.make_node("Mul", [f"Y_other_c{c}", f"one_minus_M_bottom_c{c}"], [f"Y_other_scaled_c{c}"]))
        nodes.append(helper.make_node("Add", [f"Y_bottom_c{c}", f"Y_other_scaled_c{c}"], [f"Y_c{c}"]))

        y_channels.append(f"Y_c{c}")

    # Combine Y channels to get total active mask Y_total = Max(Y_c1, ..., Y_c9)
    nodes.append(helper.make_node("Max", y_channels, ["Y_total"]))

    # B_0 = G_grid - Y_total
    nodes.append(helper.make_node("Sub", ["G_grid", "Y_total"], ["B_0"]))

    # Concat output channels
    nodes.append(helper.make_node("Concat", ["B_0"] + y_channels, ["output"], axis=1))

    # All initializers
    initializer_list = [
        slice_chan_axes, slice_h_axes, slice_w_axes,
        start_0, end_30, start_1, end_31,
        one_const, zero_const, K_left
    ] + list(starts_ends.values())

    # 4. Make Graph and Model
    graph = helper.make_graph(
        nodes,
        "shear_graph",
        [x],
        [y],
        initializer=initializer_list
    )

    model = helper.make_model(graph, ir_version=utils._IR_VERSION, opset_imports=utils._OPSET_IMPORTS)
    return model

if __name__ == "__main__":
    task_num = 4
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
