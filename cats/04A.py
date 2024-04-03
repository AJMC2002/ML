import numpy as np


class GradientOptimizer:
    def __init__(self, oracle, x0):
        self.oracle = oracle
        self.x0 = x0

    def optimize(self, iterations: int, eps: float, alpha: float) -> np.array:
        x_opt = self.x0
        for i in range(iterations):
            grad = self.oracle.get_grad(x_opt)
            if np.linalg.norm(grad, ord=2) < eps:
                break
            x_opt -= alpha * grad
        return x_opt
