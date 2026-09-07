# Recurrent Neural Network
import numpy as np


def backpropagation(x, W_xh, W_hh, y_true, b):
    n = len(x)
    h = np.zeros(n + 1)  # Initial States
    # Time Step t
    for t in range(n):
        z = (W_xh * x[t] + W_hh * h[t]) + b
        h[t + 1] = np.tanh(z)

    # Remove h0 from the hidden states
    h_states = np.array(h[1:])
    print("Hidden States:")
    for i, val in enumerate(h_states, 1):
        print(f"+ h{i} = {val}")
    # Suppose a target output
    y_pred = h_states[-1]  # y_pred is the last index
    # Backpropagation Through Time
    delta = np.zeros(n)
    # Find delta[-1]
    delta[-1] = (y_pred - y_true) * (1 - h_states[-1] ** 2)
    # Propagating Back in Time to find delta_3, delta_2, and delta_1
    # Formula: delta_t = delta_(t+1) . W_hh . (1 - h_t ** 2)
    # Here we refactor the formula
    for t in range(n-1, 0, -1):
        delta[t - 1] = delta[t] * W_hh * (1 - h_states[t - 1] ** 2)
    print("\nBackpropagation signals:")
    for i, val in enumerate(delta, 1):
        print(f"+ delta_{i} = {val}")

    # Compute total gradients
    print("\nTotal Gradients:")
    contributions_wxh = np.zeros(n)
    contributions_whh = np.zeros(n)
    contributions_b = delta

    print("(1) Gradient with respect to W_xh:")
    for t1 in range(n):
        contributions_wxh[t1] = x[t1] * delta[t1]
        print(f"+ Contribution of t{t1}: {contributions_wxh[t1]}")
    print(f"=> {np.sum(contributions_wxh)}")

    print("(2) Gradient with respect to W_hh:")
    for t2 in range(n):
        contributions_whh[t2] = h[t2] * delta[t2]
        print(f"+ Contribution of t{t2}: {contributions_whh[t2]}")
    print(f"=> {np.sum(contributions_whh)}")

    print("(3) Gradient with respect to bias:")
    # The bias gradient is simply the sum of all backpropagation signals
    print(f"=> {np.sum(contributions_b)}")


# Backpropagation Through Time (BPTT)
print("Backpropagation Through Time (BPTT)")
x1 = [0.5, 0.2, 0.9, 0.3]
print(f"x1: {x1}")
backpropagation(x1, W_xh=0.8, W_hh=0.5, y_true=1, b=0)
# Another example for demonstration
x2 = [0.5, 0.4, 0.8, 0.3, 0.4, 0.7]
print(f"\nx2: {x2}")
backpropagation(x2, W_xh=0.9, W_hh=0.3, y_true=1, b=0.05)
