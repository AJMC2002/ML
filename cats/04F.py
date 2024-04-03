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


class Adam:
    """Represents an Adam optimizer

    Fields:
        eta: learning rate
        beta1: first moment decay rate
        beta2: second moment decay rate
        epsilon: smoothing term
    """

    eta: float
    beta1: float
    beta2: float
    epsilon: float

    def __init__(
        self,
        *,
        eta: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        """Initalizes `eta`, `beta1` and `beta2` fields"""
        super().__init__()
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
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
        m = np.zeros(x0.size)
        v = np.zeros(x0.size)
        for i in range(max_iter):
            grad = oracle.gradient(x_opt)
            if np.linalg.norm(grad) < eps:
                break
            m = self.beta1 * m + (1.0 - self.beta1) * grad
            m_unbiased = m / (1.0 - self.beta1 ** (i + 1))
            v = self.beta2 * v + (1.0 - self.beta2) * grad**2
            v_unbiased = v / (1.0 - self.beta2 ** (i + 1))
            x_opt -= self.eta * m_unbiased / (np.sqrt(v_unbiased) + self.epsilon)
        return x_opt
