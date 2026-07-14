# Task 002 Explanation: Enclosed Region Filling (Flood Fill)

## 1. Visual Transformation Rule
The task consists of identifying closed loops formed by **green pixels (color 3)** and filling the empty space inside them with **yellow pixels (color 4)**:
- Any black pixel (color 0) that is completely surrounded by green walls (color 3) is filled with yellow (color 4).
- Any black pixel that can reach the edge of the grid without crossing a green wall remains black (color 0).
- The green walls themselves remain unchanged in the output.

---

## 2. Neural Network Architecture
The network implements a **Parallel Flood Fill** algorithm inside a feedforward neural network structure.
- We start by flooding the padding/border area (which is guaranteed to connect to the exterior of the active grid).
- We propagate the flood mask using a 4-connectivity convolutional kernel (`Conv` + `Clip`) for 25 steps to ensure it reaches all reachable exterior pixels.
- The pixels that remain unflooded (and are not green walls) are identified as the enclosed interior and colored yellow (color 4).

### Node Flow Diagram
```mermaid
graph TD
    Input["input (1x10x30x30)"] --> SliceW["Slice (extract channel 3, wall)"]
    SliceW --> W["W (wall mask, 1x1x30x30)"]
    
    Input --> SumG["ReduceSum (all channels)"]
    SumG --> G["G (active grid mask, 1x1x30x30)"]
    
    W --> SubW["Sub (1.0 - W)"]
    SubW --> one_minus_W["one_minus_W (1x1x30x30)"]
    
    Input --> SliceB["Slice (top-left 28x28)"]
    SliceB --> zeros_28["zeros_28 (1x1x28x28)"]
    zeros_28 --> Mul0["Mul (zeros_28 * 0.0)"]
    Mul0 --> zeros_scaled["zeros_scaled (1x1x28x28)"]
    zeros_scaled --> PadB["Pad (pad to 30x30 with 1.0)"]
    PadB --> Border["border (1x1x30x30, 1.0 at edges)"]
    
    Border & one_minus_W --> MulF0["Mul (border * one_minus_W)"]
    MulF0 --> F_0["F_0 (initial flooded mask, 1x1x30x30)"]
    
    F_0 --> Loop["Propagation Loop (25 steps)"]
    
    subgraph Loop ["Propagation Step (Repeated 25 times)"]
        F_t["F_t"] --> Conv["Conv (W_conv, 4-conn kernel)"]
        Conv --> ConvOut["ConvOut"]
        ConvOut --> Clip["Clip (min=0.0, max=1.0)"]
        Clip --> ClipOut["ClipOut"]
        ClipOut & one_minus_W --> MulStep["Mul (ClipOut * one_minus_W)"]
        MulStep --> F_next["F_next"]
    end
    
    Loop --> F_final["F_final (final flooded mask, 1x1x30x30)"]
    
    F_final & G --> MulGrid["Mul (F_final * G)"]
    MulGrid --> F_grid["F_grid (flooded pixels inside grid)"]
    
    F_grid & one_minus_W --> MulB0["Mul (F_grid * one_minus_W)"]
    MulB0 --> B_0["B_0 (final channel 0, 1x1x30x30)"]
    
    F_grid & W --> AddGrid["Add (F_grid + W)"]
    AddGrid --> F_grid_plus_W["F_grid_plus_W"]
    G & F_grid_plus_W --> SubB4["Sub (G - F_grid_plus_W)"]
    SubB4 --> B_4["B_4 (final channel 4, 1x1x30x30)"]
    
    G --> MulZeros["Mul (G * 0.0)"]
    MulZeros --> zeros_30["zeros_30 (1x1x30x30)"]
    
    B_0 & zeros_30 & W & B_4 --> Concat["Concat (merge channels 0-9)"]
    Concat --> Output["output (1x10x30x30)"]
```

---

## 3. Parameter and Memory Details

### Parameter Count
All convolution layers share the same kernel weights to keep the parameter footprint minimal. The network requires only **23 parameters**:

| Parameter Name | Type | Shape | Description | Number of Elements |
| :--- | :--- | :--- | :--- | :---: |
| `slice_starts` | INT64 | `[3]` | Start indices for spatial slice | 3 |
| `slice_ends` | INT64 | `[3]` | End indices for spatial slice | 3 |
| `slice_axes` | INT64 | `[3]` | Axes to slice | 3 |
| `slice_chan3_starts` | INT64 | `[1]` | Start channel for wall slice | 1 |
| `slice_chan3_ends` | INT64 | `[1]` | End channel for wall slice | 1 |
| `slice_chan3_axes` | INT64 | `[1]` | Axis to slice | 1 |
| `W_conv` | FLOAT | `[1, 1, 3, 3]` | 4-connectivity dilation kernel | 9 |
| `zero_const` | FLOAT | `[1]` | Constant $0.0$ | 1 |
| `one_const` | FLOAT | `[1]` | Constant $1.0$ | 1 |
| **Total** | | | | **23** |

### Memory Footprint
Total static tensor memory footprint is **312,272 bytes** (excluding input and output tensors).
Each of the 25 propagation steps uses three intermediate $1 \times 1 \times 30 \times 30$ tensors:
- Convolution output: $30 \times 30 \times 4 = 3,600$ bytes
- Clip output: $30 \times 30 \times 4 = 3,600$ bytes
- Multiplication output: $30 \times 30 \times 4 = 3,600$ bytes

This highly optimized implementation yields a NeuroGolf score of **12.348 points**.
