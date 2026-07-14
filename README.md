# The 2026 NeuroGolf Championship

Welcome to the repository for **The 2026 NeuroGolf Championship**. The objective of this competition is to design the **absolutely smallest possible neural networks** (in terms of parameter count and static memory footprint) that correctly solve ARC-AGI (Abstraction and Reasoning Corpus) image transformations.

---

## 🏆 Competition Mechanics & Rules

### The Goal
For each ARC task (represented as `taskXXX.json` containing `train`, `test`, and `arc-gen` examples), we design a minimal neural network. The input grid of integers (colors 0–9) is converted into a one-hot encoded tensor of shape `(1, 10, 30, 30)` and passed through the network. The network must output exactly a `1.0` in the correct color channel and `0.0` in all other channels to reconstruct the target grid of the same dimensions.

### The Score
Our score is calculated as:
$$\text{Score} = \max\left(1, 25 - \ln(\text{Params} + \text{Memory})\right)$$
Where:
- $\text{Params}$: The total number of parameters in the model (initializers, constants).
- $\text{Memory}$: The sum of static shapes in bytes of all intermediate outputs (activated/produced tensors) in the ONNX graph.

To maximize points, we must keep **both parameter counts and intermediate tensor sizes as small as possible**.

### Constraints
* **Format:** Models must be saved as statically-defined `.onnx` files.
* **Size Limit:** Max file size per ONNX model is **1.44 MB**.
* **Banned Operators:** The following ONNX operators are **STRICTLY BANNED**:
  * `Loop`
  * `Scan`
  * `NonZero`
  * `Unique`
  * `Script`
  * `Function`
  * `Compress`

---

## 📂 Repository Structure

```
├── .gitignore               # Excludes local datasets, venv, and profiling trace files
├── README.md                # This file (project overview and rules)
├── neurogolf_utils/         # Official NeuroGolf tournament utilities (verification & scoring)
├── task_scripts/            # Python scripts building the ONNX model for each task
│   ├── task001.py           # Build script for Task 1
│   └── task002.py           # Build script for Task 2
├── explanations/            # Detailed visual and architectural explanations of the networks
│   ├── task001_explanation.md
│   └── task002_explanation.md
├── onnx_models/             # Generated ONNX models (e.g. task001.onnx, task002.onnx)
└── tasks_data/              # Local JSON dataset files (e.g. task001.json, task002.json) [Git Ignored]
```

---

## 🚀 Progress & Leaderboard

| Task | Name / Visual Rule | Params | Memory (Bytes) | NeuroGolf Score | Explanation |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **001** | Kronecker Product Fractal Expansion | 26 | 10,800 | **15.710** | [task001_explanation.md](explanations/task001_explanation.md) |
| **002** | Enclosed Region Filling (Flood Fill) | 23 | 312,272 | **12.348** | [task002_explanation.md](explanations/task002_explanation.md) |
| **003** | Vertical Periodic Pattern Extension | 47 | 18,376 | **15.179** | [task003_explanation.md](explanations/task003_explanation.md) |

---

## 🛠️ Setup & Local Verification

1. **Initialize the virtual environment & install dependencies:**
   ```bash
   uv venv
   uv pip install onnx onnx-tool onnxruntime numpy matplotlib ipython
   ```

2. **Run and verify a task build script:**
   ```bash
   .venv/Scripts/python task_scripts/task001.py
   ```
