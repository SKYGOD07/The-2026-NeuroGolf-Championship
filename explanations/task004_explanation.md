# Task 004 Explanation: Sheared Shape Verticalization

## 1. Visual Transformation Rule
The task is to identify shapes that are sheared/leaning horizontally, and "straighten" them to make them vertical:
- For each color channel (shapes are defined by distinct colors), we want to undo the shear.
- Specifically, the bottom row of each shape acts as the "anchor" and does not shift.
- All rows above the bottom row shift to the right by 1 pixel, but they are constrained so they do not exceed the rightmost column of the shape (which is the rightmost column of the anchor row).
- In other words, we shift the pixels to the right by 1, except for pixels that are already at the rightmost boundary of the shape.
- The background color 0 (black) is reconstructed to fill in any empty space left behind by the shifted shapes.

---

## 2. Neural Network Architecture
The network is completely statically defined, with 0 learned weights and a very lightweight, parameter-free structure that operates in parallel over all 9 color channels.

### Shape Processing Pipeline (per Channel $c$):
1. **Grid & Boundary Identification:**
   - Detect the bottom row of the shape by finding the last row index $r_{bottom}$ where the shape is active: $M_{bottom\_row\_c} = V_c \times (1.0 - V_{up\_c})$, where $V_c$ is the row activity mask and $V_{up\_c}$ is $V_c$ shifted up by 1.
   - Extract the bottom row pixels $B_c$ and find the rightmost active column index $c_{right}$: $M_{rightmost\_pixel\_c} = B_c \times (1.0 - B_{left\_c})$ (where $B_{left\_c}$ is $B_c$ shifted left by 1).
   - Generate a column mask `col_mask_c` at $c_{right}$ by doing `ReduceMax` over height.
2. **Horizontal Shift Constraint:**
   - Generate a left-side column mask `M_left_col_c` that is $1.0$ for columns $\leq c_{right}$ and $0.0$ for columns $> c_{right}$. This is achieved using a 1D convolution (`Conv`) on `col_mask_c` with a constant step kernel of 1s of size 59 (with padding of 29).
3. **Blending and Clipping:**
   - Shift the shape channel `X_c` to the right by 1 using `Pad` and `Slice` to get `X_shifted_c`.
   - Compute the target row pixels using `Max` to allow shifting into the rightmost column but prevent shifting out of it:
     `Y_other_c = Max(X_c * col_mask_c, X_shifted_c * M_left_col_c)`.
   - Combine the bottom row (unshifted) and the other rows (sheared) using the bottom row mask:
     `Y_c = X_c * M_bottom_row_c + Y_other_c * (1.0 - M_bottom_row_c)`.

---

## 3. Parameter and Memory Details

### Parameter Count
The model requires only **86 parameters** (almost all of which are metadata/configuration constants like slicing indices).

| Parameter Name | Type | Shape | Description | Number of Elements |
| :--- | :--- | :--- | :--- | :---: |
| `K_left` | FLOAT | `[1, 1, 1, 59]` | 1D kernel step function for causal rightmost mask | 59 |
| Slicing & padding constants | INT64 | various | Row, column, and channel indexing metadata | 25 |
| `one_const` / `zero_const` | FLOAT | `[1]`, `[1]` | Constants $1.0$ and $0.0$ | 2 |
| **Total** | | | | **86** |

### Memory Footprint
Total static tensor memory footprint is **309,120 bytes** (excluding input and output tensors).

This highly optimized implementation yields a NeuroGolf score of **11.928 points**.
