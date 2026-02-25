import numpy as np
import matplotlib.pyplot as plt

def evaluate_poly(coefficients, x):
    """
    Evaluates the polynomial at point x using NumPy's polyval.
    Coefficients should be in descending order: [an, an-1, ..., a0]
    """
    return np.polyval(coefficients, x)

def bisection_method(coeffs, a, b, tol, max_iter):
    """
    Solves for a root within the interval [a, b].
    """
    fa = evaluate_poly(coeffs, a)
    fb = evaluate_poly(coeffs, b)

    # Validate Intermediate Value Theorem condition
    if fa * fb >= 0:
        print("\n[Error]: f(a) and f(b) must have opposite signs.")
        print(f"f(a) = {fa:.4f}, f(b) = {fb:.4f}")
        return None, None

    print(f"\n{'Iter':<5} | {'a':<10} | {'b':<10} | {'c (Root)':<10} | {'f(c)':<10} | {'Error':<10}")
    print("-" * 70)

    iteration_data = []
    c = a
    
    for i in range(1, max_iter + 1):
        prev_c = c
        c = (a + b) / 2
        fc = evaluate_poly(coeffs, c)
        
        # Calculate relative error (except for first iteration)
        error = abs(c - prev_c) if i > 1 else abs(b - a)
        
        # Store data for display
        print(f"{i:<5} | {a:<10.5f} | {b:<10.5f} | {c:<10.5f} | {fc:<10.5e} | {error:<10.5e}")
        iteration_data.append((c, fc))

        # Check for convergence
        if abs(fc) < tol or (error < tol and i > 1):
            return c, i

        # Narrow the interval
        if evaluate_poly(coeffs, a) * fc < 0:
            b = c
        else:
            a = c

    return c, max_iter

def plot_polynomial(coeffs, a, b, root):
    """Generates a graph of the function and highlights the root."""
    # Create a margin for the plot
    margin = abs(b - a) * 0.5
    x = np.linspace(a - margin, b + margin, 500)
    y = [evaluate_poly(coeffs, val) for val in x]

    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='black', linewidth=1)  # X-axis
    plt.axvline(0, color='black', linewidth=1)  # Y-axis
    plt.plot(x, y, label='f(x)', color='blue')
    
    if root is not None:
        plt.plot(root, 0, 'ro', label=f'Root ~ {root:.4f}')
        plt.annotate(f'Root: {root:.4f}', (root, 0), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Polynomial Visualization & Bisection Root")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def main():
    print("--- Polynomial Solver: Bisection Method ---")
    
    try:
        # 1. User Inputs
        degree = int(input("Enter the degree of the polynomial: "))
        coeffs = []
        for i in range(degree, -1, -1):
            coeffs.append(float(input(f"Enter coefficient for x^{i}: ")))

        a = float(input("Enter lower bound (a): "))
        b = float(input("Enter upper bound (b): "))
        tol = float(input("Enter tolerance (e.g., 0.0001): "))
        max_iter = int(input("Enter maximum iterations: "))

        # 2. Run Algorithm
        root, iters = bisection_method(coeffs, a, b, tol, max_iter)

        # 3. Final Results
        if root is not None:
            print("-" * 70)
            print(f"Convergence Status: {'Success' if iters < max_iter else 'Max Iterations Reached'}")
            print(f"Approximate Root: {root:.6f}")
            print(f"Iterations Used:  {iters}")
            
            # 4. Visualization
            plot_polynomial(coeffs, a, b, root)

    except ValueError:
        print("\n[Input Error]: Please enter valid numerical values.")

if __name__ == "__main__":
    main()