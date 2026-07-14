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
    slice_chan_axes = helper.make_tensor("slice_chan_axes", TensorProto.INT64, [1], [1])
    axes_hw = helper.make_tensor("axes_hw", TensorProto.INT64, [2], [2, 3])

    one_const = helper.make_tensor("one_const", TensorProto.FLOAT, [1], [1.0])
    zero_const = helper.make_tensor("zero_const", TensorProto.FLOAT, [1], [0.0])
    half_const = helper.make_tensor("half_const", TensorProto.FLOAT, [1], [0.5])

    # 3x3 convolution kernel of ones
    conv_kernel = helper.make_tensor("conv_kernel", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9)

    # channel slicing starts/ends for each channel 1..9
    starts_ends = {}
    for c in range(1, 10):
        starts_ends[f"start_c{c}"] = helper.make_tensor(f"start_c{c}", TensorProto.INT64, [1], [c])
        starts_ends[f"end_c{c}"] = helper.make_tensor(f"end_c{c}", TensorProto.INT64, [1], [c+1])

    # Define candidate vectors for shifts of step size 4
    CANDIDATES = [
        (0, 4), (0, -4), (4, 0), (-4, 0),
        (4, 4), (-4, -4), (-4, 4), (4, -4)
    ]

    # Pre-create Starts/Ends initializers for all shifts needed
    shifts_initializers = {}
    
    # We need shifts for k*s (k up to 6)
    all_shifts = set()
    for dy, dx in CANDIDATES:
        for k in range(1, 7):
            s_dy, s_dx = k * dy, k * dx
            if abs(s_dy) < 30 and abs(s_dx) < 30:
                all_shifts.add((s_dy, s_dx))

    for dy, dx in all_shifts:
        h_start = max(0, -dy)
        w_start = max(0, -dx)
        shifts_initializers[f"starts_{dy}_{dx}"] = helper.make_tensor(f"starts_{dy}_{dx}", TensorProto.INT64, [2], [h_start, w_start])
        shifts_initializers[f"ends_{dy}_{dx}"] = helper.make_tensor(f"ends_{dy}_{dx}", TensorProto.INT64, [2], [h_start+30, w_start+30])

    nodes = []

    # Get active grid mask G_grid = sum over all 10 channels of input
    nodes.append(helper.make_node("ReduceSum", ["input"], ["G_grid"], axes=[1], keepdims=1))

    # Helper function to shift tensor
    def shift_tensor(tensor_in, dy, dx, name_prefix):
        pad_top = max(0, dy)
        pad_bottom = max(0, -dy)
        pad_left = max(0, dx)
        pad_right = max(0, -dx)
        
        padded = f"{name_prefix}_padded"
        nodes.append(helper.make_node("Pad", [tensor_in], [padded], pads=[0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right], value=0.0))
        
        shifted = f"{name_prefix}_shifted"
        nodes.append(helper.make_node("Slice", [
            padded, 
            f"starts_{dy}_{dx}", f"ends_{dy}_{dx}", 
            "axes_hw"
        ], [shifted]))
        return shifted

    # a. Slice input into X_c1 to X_c9
    for c in range(1, 10):
        nodes.append(helper.make_node("Slice", ["input", f"start_c{c}", f"end_c{c}", "slice_chan_axes"], [f"X_c{c}"]))

    # b. Compute local density for each channel c
    for c in range(1, 10):
        nodes.append(helper.make_node("Conv", [f"X_c{c}", "conv_kernel"], [f"density_map_{c}"], pads=[1, 1, 1, 1]))
        nodes.append(helper.make_node("ReduceMax", [f"density_map_{c}"], [f"density_{c}"], axes=[1, 2, 3], keepdims=1))

    # c. max_density = Max(density_1, ..., density_9)
    nodes.append(helper.make_node("Max", [f"density_{c}" for c in range(1, 10)], ["max_density"]))

    # d. is_template_c = Sub(1.0, Cast(Less(density_c, max_density)))
    for c in range(1, 10):
        nodes.append(helper.make_node("Less", [f"density_{c}", "max_density"], [f"is_less_bool_{c}"]))
        nodes.append(helper.make_node("Cast", [f"is_less_bool_{c}"], [f"is_less_{c}"], to=TensorProto.FLOAT))
        nodes.append(helper.make_node("Sub", ["one_const", f"is_less_{c}"], [f"is_template_c{c}"]))

    # e. Get template shape: Template_Shape = Max(X_c * is_template_c)
    for c in range(1, 10):
        nodes.append(helper.make_node("Mul", [f"X_c{c}", f"is_template_c{c}"], [f"Template_Shape_c{c}"]))
    nodes.append(helper.make_node("Max", [f"Template_Shape_c{c}" for c in range(1, 10)], ["Template_Shape"]))

    # f. Compute shifts of Template_Shape for all k*s
    for s_idx, (dy, dx) in enumerate(CANDIDATES):
        for k in range(1, 7):
            k_dy, k_dx = k * dy, k * dx
            if abs(k_dy) < 30 and abs(k_dx) < 30:
                shift_tensor("Template_Shape", k_dy, k_dx, f"Template_k{k}_s{s_idx}")

    # g. Compute overlaps and determine replication for each channel c
    y_channels = []
    for c in range(1, 10):
        term_names = []
        for s_idx, (dy, dx) in enumerate(CANDIDATES):
            # overlap = ReduceSum(X_c * Template_k1_s_shifted)
            # Since Template_k1_s_shifted is shifted by dy, dx, it represents Template_Shape shifted by s.
            k1_shifted = f"Template_k1_s{s_idx}_shifted"
            
            nodes.append(helper.make_node("Mul", [f"X_c{c}", k1_shifted], [f"overlap_mul_s{s_idx}_c{c}"]))
            nodes.append(helper.make_node("ReduceSum", [f"overlap_mul_s{s_idx}_c{c}"], [f"overlap_s{s_idx}_c{c}"], axes=[1, 2, 3], keepdims=1))

            # is_shift = Cast(Greater(overlap_s_idx_c, half_const))
            nodes.append(helper.make_node("Greater", [f"overlap_s{s_idx}_c{c}", "half_const"], [f"is_shift_bool_s{s_idx}_c{c}"]))
            nodes.append(helper.make_node("Cast", [f"is_shift_bool_s{s_idx}_c{c}"], [f"is_shift_s{s_idx}_c{c}"], to=TensorProto.FLOAT))

            # replicated_s = Max(Template_k1_s_shifted, ..., Template_kN_s_shifted)
            k_shifts = []
            for k in range(1, 7):
                k_dy, k_dx = k * dy, k * dx
                if abs(k_dy) < 30 and abs(k_dx) < 30:
                    k_shifts.append(f"Template_k{k}_s{s_idx}_shifted")

            if k_shifts:
                replicated_s = f"replicated_s{s_idx}_c{c}"
                nodes.append(helper.make_node("Max", k_shifts, [replicated_s]))

                term_s = f"term_s{s_idx}_c{c}"
                nodes.append(helper.make_node("Mul", [replicated_s, f"is_shift_s{s_idx}_c{c}"], [term_s]))
                term_names.append(term_s)

        if term_names:
            nodes.append(helper.make_node("Max", term_names, [f"Y_replicated_c{c}"]))
        else:
            nodes.append(helper.make_node("Mul", [f"X_c{c}", "zero_const"], [f"Y_replicated_c{c}"]))

        # Y_c = X_c * is_template_c + Y_replicated_c * (1.0 - is_template_c)
        nodes.append(helper.make_node("Sub", ["one_const", f"is_template_c{c}"], [f"one_minus_is_template_c{c}"]))
        nodes.append(helper.make_node("Mul", [f"X_c{c}", f"is_template_c{c}"], [f"Y_c_temp{c}"]))
        nodes.append(helper.make_node("Mul", [f"Y_replicated_c{c}", f"one_minus_is_template_c{c}"], [f"Y_c_rep_scaled{c}"]))
        nodes.append(helper.make_node("Add", [f"Y_c_temp{c}", f"Y_c_rep_scaled{c}"], [f"Y_c{c}"]))
        y_channels.append(f"Y_c{c}")

    # Combine Y channels to get total active mask Y_total = Max(Y_c1, ..., Y_c9)
    nodes.append(helper.make_node("Max", y_channels, ["Y_total"]))

    # B_0 = G_grid - Y_total
    nodes.append(helper.make_node("Sub", ["G_grid", "Y_total"], ["B_0"]))

    # Concat output channels and mask to active grid
    nodes.append(helper.make_node("Concat", ["B_0"] + y_channels, ["output_raw"], axis=1))
    nodes.append(helper.make_node("Mul", ["output_raw", "G_grid"], ["output"]))

    # All initializers
    initializer_list = [
        slice_chan_axes, axes_hw,
        one_const, zero_const, half_const, conv_kernel
    ] + list(starts_ends.values()) + list(shifts_initializers.values())

    # 4. Make Graph and Model
    graph = helper.make_graph(
        nodes,
        "replication_graph",
        [x],
        [y],
        initializer=initializer_list
    )

    model = helper.make_model(graph, ir_version=utils._IR_VERSION, opset_imports=utils._OPSET_IMPORTS)
    return model

if __name__ == "__main__":
    task_num = 5
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
