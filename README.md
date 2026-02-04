# Multi-Fidelity-DNNs-PINNs
Multi-fidelity DNNs for function approximation. This repo demonstrates a simple multi-fidelity function approximation setup where a low-fidelity (LF) network assists a high-fidelity (HF) network via linear and nonlinear correlations. The training first fits LF on more abundant data, then conditions HF on the LF predictions at sparse HF locations.

Reference concept: `https://www.sciencedirect.com/science/article/pii/S0021999119307260`

## Quickstart (Windows PowerShell)

```bash
cd .
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python mf_func.py
```

## Environment & Dependencies

- Python 3.8+ (tested on Windows)
- PyTorch 2.x
- Numpy, Matplotlib, SciPy

Install via the pinned `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Project Structure

- `mf_func.py`: Full training script using LF and HF networks, Adam optimizer, plotting and saving `linear_func.png`.
- `net.py`: Lightweight MLP utilities (weights init, forward pass) used by earlier version; current script performs its own init.
- `run_minimal.py`: Minimal self-contained example (see below) mirroring the training and plotting pipeline.
- `requirements.txt`: Pinned dependencies for reproducibility.

## Minimal Script (run_minimal.py)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(1234)
torch.manual_seed(1234)

# Low-/High-fidelity ground-truth functions
def fun_lf(x):
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

def fun_hf(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

# Simple fully-connected network with tanh
class FNN(nn.Module):
    def __init__(self, layers, Xmin, Xmax):
        super(FNN, self).__init__()
        self.layers = layers
        self.Xmin = torch.tensor(Xmin, dtype=torch.float32)
        self.Xmax = torch.tensor(Xmax, dtype=torch.float32)
        self.network = nn.Sequential()
        
        # Build the network
        for i in range(len(layers) - 1):
            self.network.add_module(f'linear{i}', nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # Add tanh activation except for last layer
                self.network.add_module(f'tanh{i}', nn.Tanh())
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                in_dim = m.in_features
                out_dim = m.out_features
                std = np.sqrt(2.0 / (in_dim + out_dim))
                nn.init.normal_(m.weight, mean=0, std=std)
                nn.init.zeros_(m.bias)
    
    def forward(self, X):
        # Normalize input
        A = 2.0 * (X - self.Xmin) / (self.Xmax - self.Xmin) - 1.0
        return self.network(A)

def main():
    D = 1
    # Architectures
    layers_lf = [D, 20, 20, 1]
    layers_hf_nl = [D + 1, 10, 10, 1]
    layers_hf_l = [D + 1, 1]

    # Data
    x_lf = np.linspace(0, 1, 21).reshape((-1, 1))
    y_lf = fun_lf(x_lf)
    x_hf = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))
    y_hf = fun_hf(x_hf)

    Xmin, Xmax = x_lf.min(0), x_lf.max(0)
    Ymin, Ymax = y_lf.min(0), y_lf.max(0)
    Xhmin = np.hstack((Xmin, Ymin))
    Xhmax = np.hstack((Xmax, Ymax))

    # Convert data to torch tensors
    x_lf_tensor = torch.tensor(x_lf, dtype=torch.float32)
    y_lf_tensor = torch.tensor(y_lf, dtype=torch.float32)
    x_hf_tensor = torch.tensor(x_hf, dtype=torch.float32)
    y_hf_tensor = torch.tensor(y_hf, dtype=torch.float32)

    # Networks
    model_lf = FNN(layers_lf, Xmin, Xmax)
    model_hf_nl = FNN(layers_hf_nl, Xhmin, Xhmax)
    model_hf_l = FNN(layers_hf_l, Xhmin, Xhmax)

    # Optimizer
    params = list(model_lf.parameters()) + list(model_hf_nl.parameters()) + list(model_hf_l.parameters())
    optimizer = optim.Adam(params, lr=1e-3)

    # Loss function
    def compute_loss():
        # Forward pass for low-fidelity
        y_pred_lf = model_lf(x_lf_tensor)
        loss_lf = torch.mean((y_pred_lf - y_lf_tensor) ** 2)
        
        # Forward pass for high-fidelity
        y_pred_lf_hf = model_lf(x_hf_tensor)
        hf_in = torch.cat([x_hf_tensor, y_pred_lf_hf], dim=-1)
        y_pred_hf_nl = model_hf_nl(hf_in)
        y_pred_hf_l = model_hf_l(hf_in)
        y_pred_hf = y_pred_hf_l + y_pred_hf_nl
        loss_hf = torch.mean((y_pred_hf - y_hf_tensor) ** 2)
        
        # Regularization
        reg = 0.01 * sum(torch.sum(w ** 2) for w in model_hf_nl.parameters())
        
        return loss_lf + loss_hf + reg, loss_lf, loss_hf

    # Train
    nmax, loss_c, cur, n = 5000, 1e-3, 1.0, 0
    while n < nmax and cur > loss_c:
        n += 1
        optimizer.zero_grad()
        loss, loss_lf, loss_hf = compute_loss()
        loss.backward()
        optimizer.step()
        
        cur = loss.item()
        if n % 1000 == 0:
            print('n: %d, loss: %.3e, loss_lf: %.3e, loss_hf: %.3e' % (n, cur, loss_lf.item(), loss_hf.item()))

    # Predict and plot
    x_test = np.linspace(0, 1, 1000).reshape((-1, 1))
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    # Predict low-fidelity
    model_lf.eval()
    with torch.no_grad():
        y_lf_test = model_lf(x_test_tensor).numpy()
        
        # Predict high-fidelity
        y_pred_lf_hf_test = model_lf(x_test_tensor)
        hf_in_test = torch.cat([x_test_tensor, y_pred_lf_hf_test], dim=-1)
        y_pred_hf_nl_test = model_hf_nl(hf_in_test)
        y_pred_hf_l_test = model_hf_l(hf_in_test)
        y_hf_test = (y_pred_hf_l_test + y_pred_hf_nl_test).numpy()
    
    y_lf_ref = fun_lf(x_test)
    y_hf_ref = fun_hf(x_test)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_lf, y_lf, 'go', label='LF train')
    plt.plot(x_test, y_lf_ref, 'k-', label='LF true')
    plt.plot(x_test, y_lf_test, 'r--', label='LF pred')
    plt.plot(x_hf, y_hf, 'bo', label='HF train')
    plt.plot(x_test, y_hf_ref, 'k-', label='HF true')
    plt.plot(x_test, y_hf_test, 'c--', label='HF pred')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multi-Fidelity Function Approximation (Adam only)')
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig('linear_func.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
```

## Results

- The script prints training progress every 1000 steps and saves the figure `linear_func.png` showing LF/HF training points, true curves, and predictions with a legend.

## Notes & FAQ

- This repository uses PyTorch 2.x for implementation, which provides better performance and easier installation compared to TensorFlow.
- CPU/GPU: PyTorch automatically uses GPU if available, otherwise falls back to CPU.

## Citation


If you use this repository, please cite the original multi-fidelity DNN concept paper linked above.

