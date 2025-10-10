# Multi-Fidelity-DNNs-PINNs
Multi-fidelity DNNs for function approximation. This repo demonstrates a simple multi-fidelity function approximation setup where a low-fidelity (LF) network assists a high-fidelity (HF) network via linear and nonlinear correlations. The training first fits LF on more abundant data, then conditions HF on the LF predictions at sparse HF locations.

Reference concept: `https://www.sciencedirect.com/science/article/pii/S0021999119307260`

## Quickstart (Windows PowerShell)

```bash
# 1) 进入项目目录
cd "C:\Users\17868\OneDrive\Desktop\DL\Multi-Fidelity-DNNs-PINNs\Multi-Fidelity-DNNs-PINNs"

# 2) 创建并激活虚拟环境
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 3) 安装依赖（使用已固定版本）
pip install -r requirements.txt

# 4) 运行脚本（保存 PNG 到 linear_func.png）
python mf_func.py
```

可选：若要运行一个最小示例，将如下脚本保存为 `run_minimal.py` 后执行 `python run_minimal.py`。

## Environment & Dependencies

- Python 3.8+ (tested on Windows)
- TensorFlow 2.x with v1-compat (we use `tensorflow.compat.v1` and disable v2 behavior)
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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
tf.set_random_seed(1234)

# Low-/High-fidelity ground-truth functions
def fun_lf(x):
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

def fun_hf(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

# Simple fully-connected network with tanh
def init_weights(layers):
    W, b = [], []
    for in_dim, out_dim in zip(layers[:-1], layers[1:]):
        std = np.sqrt(2.0 / (in_dim + out_dim))
        W.append(tf.Variable(tf.random.normal([in_dim, out_dim], stddev=std)))
        b.append(tf.Variable(tf.zeros([1, out_dim])))
    return W, b

def fnn(W, b, X, Xmin, Xmax):
    A = 2.0 * (X - Xmin) / (Xmax - Xmin) - 1.0
    for Wi, bi in zip(W[:-1], b[:-1]):
        A = tf.tanh(tf.matmul(A, Wi) + bi)
    return tf.matmul(A, W[-1]) + b[-1]

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

    # Placeholders
    x_train_lf = tf.placeholder(tf.float32, shape=[None, D])
    y_train_lf = tf.placeholder(tf.float32, shape=[None, 1])
    x_train_hf = tf.placeholder(tf.float32, shape=[None, D])
    y_train_hf = tf.placeholder(tf.float32, shape=[None, 1])

    # Networks
    W_lf, b_lf = init_weights(layers_lf)
    W_hf_nl, b_hf_nl = init_weights(layers_hf_nl)
    W_hf_l, b_hf_l = init_weights(layers_hf_l)

    # Forward
    y_pred_lf = fnn(W_lf, b_lf, x_train_lf, Xmin, Xmax)
    y_pred_lf_hf = fnn(W_lf, b_lf, x_train_hf, Xmin, Xmax)
    hf_in = tf.concat([x_train_hf, y_pred_lf_hf], axis=-1)
    y_pred_hf_nl = fnn(W_hf_nl, b_hf_nl, hf_in, Xhmin, Xhmax)
    y_pred_hf_l = fnn(W_hf_l, b_hf_l, hf_in, Xhmin, Xhmax)
    y_pred_hf = y_pred_hf_l + y_pred_hf_nl

    # Loss and optimizer (Adam only, TF2-compatible)
    reg = 0.01 * tf.add_n([tf.nn.l2_loss(w_) for w_ in W_hf_nl])
    loss_lf = tf.reduce_mean(tf.square(y_pred_lf - y_train_lf))
    loss_hf = tf.reduce_mean(tf.square(y_pred_hf - y_train_hf))
    loss = loss_lf + loss_hf + reg
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # Train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    feed = {x_train_lf: x_lf, y_train_lf: y_lf, x_train_hf: x_hf, y_train_hf: y_hf}
    nmax, loss_c, cur, n = 5000, 1e-3, 1.0, 0
    while n < nmax and cur > loss_c:
        n += 1
        cur, _, lf_, hf_ = sess.run([loss, train_op, loss_lf, loss_hf], feed_dict=feed)
        if n % 1000 == 0:
            print('n: %d, loss: %.3e, loss_lf: %.3e, loss_hf: %.3e' % (n, cur, lf_, hf_))

    # Predict and plot
    x_test = np.linspace(0, 1, 1000).reshape((-1, 1))
    y_lf_ref = fun_lf(x_test)
    y_hf_ref = fun_hf(x_test)
    y_lf_test = sess.run(y_pred_lf, feed_dict={x_train_lf: x_test})
    y_hf_test = sess.run(y_pred_hf, feed_dict={x_train_hf: x_test})

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

- This repository uses TensorFlow 2.x runtime with v1 compatibility (`tf.compat.v1.disable_v2_behavior`). If you want an LBFGS stage similar to classic TF1 `tensorflow.contrib.opt.ScipyOptimizerInterface`, you can switch to `tensorflow_probability`'s L-BFGS or SciPy wrappers.
- CPU/GPU: The script disables GPU by default via `CUDA_VISIBLE_DEVICES='-1'` for portability.

## Citation

If you use this repository, please cite the original multi-fidelity DNN concept paper linked above.