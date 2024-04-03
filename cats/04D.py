import numpy as np
import numpy.typing as npt


class Oracle:
    """Provides an interface for evaluating a function and its derivative at arbitrary point"""

    def value(self, x: npt.NDArray[np.float64]) -> float:
        """Evaluates the underlying function at point `x`

        Args:
            x: a point to evaluate funciton at

        Returns:
            Function value
        """
        raise NotImplementedError()

    def gradient(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluates the underlying function derivative at point `x`

        Args:
            x: a point to evaluate derivative at

        Returns:
            Function derivative
        """
        raise NotImplementedError()


class AdaGrad:
    """Represents an AdaGrad optimizer

    Fields:
        eta: learning rate
        epsilon: smoothing term
    """

    eta: float
    epsilon: float

    def __init__(self, *, eta: float = 0.1, epsilon: float = 1e-8):
        """Initalizes `eta` and `alpha` fields"""
        super().__init__()
        self.eta = eta
        self.epsilon = epsilon

    def optimize(
        self,
        oracle: Oracle,
        x0: npt.NDArray[np.float64],
        *,
        max_iter: int = 100,
        eps: float = 1e-5,
    ) -> npt.NDArray[np.float64]:
        """Optimizes a function specified as `oracle` starting from point `x0`.
        The optimizations stops when `max_iter` iterations were completed or
        the L2-norm of the gradient at current point is less than `eps`

        Args:
            oracle: function to optimize
            x0: point to start from
            max_iter: maximal number of iterations
            eps: threshold for L2-norm of gradient

        Returns:
            A point at which the optimization stopped
        """
        x_opt = x0
        v_upd = np.zeros(x0.size)
        for i in range(max_iter):
            grad = oracle.gradient(x_opt)
            grad_norm = np.linalg.norm(grad, ord=2)
            if grad_norm < eps:
                break
            v_upd += grad**2
            x_opt -= self.eta * grad / np.sqrt(v_upd + self.epsilon)
        return x_opt
