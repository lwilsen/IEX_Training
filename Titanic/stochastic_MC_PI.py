import numpy as np


def estimate_pi(num_samples):
    count_inside_circle = 0
    for _ in range(num_samples):
        x, y = np.random.rand(2)
        if x**2 + y**2 <= 1:
            count_inside_circle += 1
    return (4 * count_inside_circle) / num_samples


pi_estimate = estimate_pi(1000000)
print(f"Estimated value of π: {pi_estimate}")
