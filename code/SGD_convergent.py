"""
SGD_convergent.py

Implements stochastic gradient descent with a diminishing
learning rate, to guarantee convergence to the global optimum
on convex functions. We follow Bottou (2012), "Stochastic
gradient descent tricks"
<www.microsoft.com/en-us/research/publication/stochastic-gradient-tricks/>

Mostly a simplified form of PyTorch's torch.optim.SGD

Author: Arya D. McCarthy <arya@jhu.edu> 2020-10-11
"""
from typing import Final, Iterable

import torch
from torch.optim.optimizer import Optimizer  # type: ignore[import]


class ConvergentSGD(Optimizer):
    """Minimize a function by stepping down the gradient """

    def __init__(self, params: Iterable[torch.Tensor], gamma0: float, lambda_: float):
        # Validate inputs.
        if gamma0 < 0.0:
            raise ValueError(f"Invalid initial learning rate: {gamma0}")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid learning rate shrinkage constant: {lambda_}")

        super().__init__(params, {})
        self.gamma0: Final[float] = gamma0  # Initial learning rate (from Algorithm 1)
        self.lambda_: Final[
            float
        ] = lambda_  # Shriking the LR coefficient (from Algorithm 1)
        self.t: int = 0  # Current time step (from Algorithm 1)

    @property
    def gamma(self) -> float:
        """Compute the current learning rate γ according to Algorithm 6, line 1."""
        gamma = self.gamma0 / (1 + self.gamma0 * self.lambda_ * self.t)
        return gamma

    @torch.no_grad()  # Don't bother with gradient bookkeeping here.
    def step(self, closure=None) -> None:
        """Perform a single optimization step, then update t."""
        # Loop over the parameters and update them (line 7 of Algorithm 1)
        gamma = self.gamma  # Cache this current value.
        for group in self.param_groups:
            for theta_i in group["params"]:
                # Skip updating parameters which lack computed gradients.
                if theta_i.grad is None:
                    continue
                d_theta = theta_i.grad  # The gradient of the objective to minimize.
                # sub_ is in-place subtraction.
                theta_i.sub_(gamma * d_theta)  # θ = θ - γ · ∇J_i(θ)

        self.t += 1  # Line 8 of hw-lm.pdf Algorithm 1. Required for diminishing the LR.


def test_me():
    model = torch.nn.Linear(2, 3)  # Generic, simple model with few parameters: f(x) = Ax+b
    x = torch.randn(2)  # Generic, simple input: random numbers.
    optimizer = ConvergentSGD(
        model.parameters(), gamma0=0.5, lambda_=20
    )  # arbitrarily chosen values
    for i in range(10):
        value = model(x).sum()
        print(value)  # If everything is working, these values should be getting lower.
        value.backward()
        optimizer.step()


if __name__ == "__main__":
    test_me()
