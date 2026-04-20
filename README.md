# 🧠 Self-Pruning Neural Network

A PyTorch implementation of a neural network that **learns to prune 
itself during training** using learnable gate parameters and L1 
sparsity regularization — trained on the CIFAR-10 dataset.

Built as part of the **Tredence Analytics AI Engineering Internship 
Case Study**.

---

## 📌 What is This?

Most neural networks are pruned **after** training. This project takes 
a different approach — the network **prunes itself during training** by 
learning which weights are unnecessary.

Each weight has a learnable "gate" (value between 0 and 1):
- Gate → **1** = weight is active and important
- Gate → **0** = weight is pruned and has no effect

An **L1 sparsity penalty** in the loss function pushes these gates 
toward zero, forcing the network to become sparse automatically.

---

## 🗂️ Project Structure

self-pruning-neural-network/
├── solution.ipynb          # Full implementation in Colab notebook
├── report.md               # Analysis report with results table
├── gate_distribution.png   # Plot of final gate value distribution
└── README.md               # You are here

---

## 🔧 How It Works

### 1. PrunableLinear Layer
A custom replacement for `torch.nn.Linear` that includes:
- Standard `weight` and `bias` parameters
- Extra `gate_scores` parameter (same shape as weights)
- Forward pass multiplies weights by `sigmoid(gate_scores)`

```python
gates = torch.sigmoid(self.gate_scores)
pruned_weights = self.weight * gates
output = F.linear(x, pruned_weights, self.bias)
```

### 2. Custom Loss Function
Total Loss = CrossEntropyLoss + λ × SparsityLoss
Where `SparsityLoss` = sum of all gate values across all layers (L1 norm)

### 3. Network Architecture
Input (3072) → FC1 (512) → FC2 (256) → FC3 (128) → Output (10)

All fully connected layers use `PrunableLinear`.

---

## 📊 Results

> Sparsity is measured as % of weights with gate value < 0.5
> (gates below 0.5 contribute less than half their original weight)

| Lambda (λ) | Test Accuracy | Sparsity (gate < 0.5) | Sparsity (gate < 0.3) |
|------------|--------------|----------------------|----------------------|
| 0.0001     | 55.85%       | 99.96%               | 0.00%                |
| 0.001      | 56.44%       | 100.00%              | 99.98%               |
| 0.01       | 55.40%       | 100.00%              | 100.00%              |

### Key Observations
- Higher λ pushes gate values lower (more aggressive pruning)
- λ = 0.01 pushed ALL gates below 0.1 — maximum sparsity
- λ = 0.001 gave best accuracy with strong sparsity
- Accuracy stays stable (~55-56%) showing the network
  adapts well even when heavily pruned

### Gate Distribution Plot
![Gate Distribution](gate_distribution.png)

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open `solution.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order

### Option 2: Local Setup
```bash
# Clone the repo
git clone https://github.com/Param-Maheshwari/self-pruning-neural-network.git
cd self-pruning-neural-network

# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the notebook
jupyter notebook solution.ipynb
```

---

## 📦 Dependencies
torch
torchvision
matplotlib
numpy

---

## 💡 Key Concepts

**Why L1 and not L2 for sparsity?**
L1 applies constant gradient pressure toward zero, which can drive 
values to *exactly* zero. L2 only drives values *close* to zero but 
rarely reaches it. This makes L1 the ideal choice for pruning.

**Why Sigmoid for gates?**
Sigmoid squashes any value into (0, 1), which perfectly represents 
"how active" a weight should be. Values near 0 mean pruned, values 
near 1 mean active.

---

## 👤 Author

**Param Maheshwari**  
[GitHub](https://github.com/Param-Maheshwari)

---
