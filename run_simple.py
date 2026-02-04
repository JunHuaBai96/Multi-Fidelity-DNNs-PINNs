import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

# Low-/High-fidelity ground-truth functions
def fun_lf(x):
    return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

def fun_hf(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

# Polynomial approximation for demonstration
def polynomial_approximation(x, y, degree):
    coeffs = np.polyfit(x.flatten(), y.flatten(), degree)
    poly = np.poly1d(coeffs)
    return poly

def main():
    # Data
    x_lf = np.linspace(0, 1, 21).reshape((-1, 1))
    y_lf = fun_lf(x_lf)
    x_hf = np.array([0.0, 0.4, 0.6, 1.0]).reshape((-1, 1))
    y_hf = fun_hf(x_hf)

    # Simple polynomial approximations for demonstration
    poly_lf = polynomial_approximation(x_lf, y_lf, 10)
    poly_hf = polynomial_approximation(x_hf, y_hf, 3)

    # Predict and plot
    x_test = np.linspace(0, 1, 1000).reshape((-1, 1))
    y_lf_ref = fun_lf(x_test)
    y_hf_ref = fun_hf(x_test)
    y_lf_test = poly_lf(x_test.flatten()).reshape((-1, 1))
    y_hf_test = poly_hf(x_test.flatten()).reshape((-1, 1))

    plt.figure(figsize=(7, 4.5))
    plt.plot(x_lf, y_lf, 'go', label='LF train')
    plt.plot(x_test, y_lf_ref, 'k-', label='LF true')
    plt.plot(x_test, y_lf_test, 'r--', label='LF pred')
    plt.plot(x_hf, y_hf, 'bo', label='HF train')
    plt.plot(x_test, y_hf_ref, 'k-', label='HF true')
    plt.plot(x_test, y_hf_test, 'c--', label='HF pred')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Multi-Fidelity Function Approximation (Polynomial)')
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig('linear_func.png', dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()