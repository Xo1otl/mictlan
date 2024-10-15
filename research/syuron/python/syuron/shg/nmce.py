import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, List


class NMCE:
    def __init__(self, kappa: Callable[[float], complex], twodelta: Callable[[], float]):
        """
        Nonlinear Medium Conversion Efficiency (NMCE) class for SHG.

        Parameters:
        kappa (Callable[[float], complex]): A function representing the nonlinear coupling constant as a function of z.
        twodelta (Callable[[], float]): A function returning the value of 2*Delta.
        """
        self.kappa = kappa
        self.twodelta = twodelta

    def solve(self, L: float, A0: complex, B0: complex) -> Tuple[complex, complex]:
        """
        Solve the coupled differential equations for A(z) and B(z).

        Parameters:
        L (float): The range of integration.
        A0 (complex): Initial condition for A(0).
        B0 (complex): Initial condition for B(0).

        Returns:
        Tuple[complex, complex]: A tuple containing B(L) and A(0) after solving the equations.
        """
        # Define the right-hand side of the differential equations
        def equations(z: float, y: List[complex]) -> List[complex]:
            A, B = y[0], y[1]
            dA_dz = -1j * np.conj(self.kappa(z)) * np.conj(A) * B * \
                np.exp(1j * self.twodelta() * z)
            dB_dz = -1j * self.kappa(z) * A**2 * \
                np.exp(1j * self.twodelta() * z)
            return [dA_dz, dB_dz]

        # Initial conditions
        y0 = [A0, B0]

        # Numerical solution using solve_ivp
        sol = solve_ivp(equations, [0, L], y0, method='RK45')

        # Return B(L) value (nearest point to L)
        B_L = sol.y[1][-1]  # Corresponding part for B(z)
        return B_L, sol.y[0]  # Return A(z) as well for visualization


def efficiency(B_L, A0):
    return abs(B_L)**2 / abs(A0)**2
