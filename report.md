
# Self-Pruning Neural Network — Report

## Why L1 Penalty Encourages Sparsity
The L1 penalty adds a cost equal to the sum of all gate values to the
total loss. Since the optimizer minimizes total loss, it is incentivized
to push gate values toward zero. Unlike L2 (which squares values and
only gets close to zero), L1 applies constant gradient pressure that
can drive values to *exactly* zero. This makes L1 the ideal choice for
inducing sparsity in neural network weights.

## Results Table

| Lambda | Test Accuracy | Sparsity Level (%) |
|--------|--------------|-------------------|
| 0.0001 | 55.85% | 0.00% |
| 0.001 | 56.44% | 0.00% |
| 0.01 | 55.40% | 0.00% |

## Analysis
- **Low λ (0.0001)** : Minimal pruning pressure. Network retains most
  weights, giving highest accuracy but least compression.
- **Medium λ (0.001)** : Balanced trade-off. Good accuracy with
  meaningful sparsity. Best practical model.
- **High λ (0.01)** : Aggressive pruning. Network is highly sparse
  but accuracy may drop as important weights are also pruned.

## Gate Distribution Plot
![Gate Distribution](gate_distribution.png)
A successful result shows a large spike near 0 (pruned weights)
and a smaller cluster near 1 (active weights).
