import numpy as np
from scipy.interpolate import interp1d
from . import *


class Grating:
    def __init__(self, widths, P0, kappa, T, resolution=5000):
        """
        Initialize the Grating class with polarization reversal widths and other parameters.

        Parameters:
            widths (np.array): Array of polarization reversal widths.
            P0 (float): Initial power |A0|^2.
            kappa (float): Coupling coefficient.
            T (float): Temperature (°C).
            Lambda_0 (float): Reference wavelength.
            resolution (int): Resolution for PeriodicObserver.
        """
        self.widths = widths
        self.P0 = P0
        self.kappa = kappa
        self.L = np.sum(widths)
        self.A0 = np.sqrt(P0)
        self.T = T
        self.Lambda_0 = widths[0]   # Reference wavelength
        self.K0 = 2 * np.pi / self.Lambda_0

        # Initialize PeriodicObserver
        self.resolution = resolution
        # dzはL/resolutionになる
        self.observer = PeriodicObserver(
            resolution, self.L, period=self.Lambda_0)

    def infer_coordinate_transformation(self):
        """
        Infer the coordinate transformation from the widths array.

        Returns:
            z_transform (function): Interpolated function z(z_prime).
        """
        # Create z(z_prime) using PeriodicObserver
        z_prime = np.linspace(0, self.L, self.resolution)
        # FIXME: self.widthsは幅配列なので中央差分による微分の配列とは異なるから修正が必要
        z_transform_discrete = self.observer.infer_transform(self.widths)

        # Interpolate to get z(z_prime)
        z_transform = interp1d(z_prime, z_transform_discrete)

        return z_transform

    def calculate_efficiency(self, lambda_val):
        """
        Calculate the conversion efficiency for a given wavelength.

        Parameters:
            lambda_val (float): Wavelength (μm).

        Returns:
            efficiency_val (float): Conversion efficiency.
        """
        # Infer the coordinate transformation
        z_transform = self.infer_coordinate_transformation()

        # Define Phi_z based on the coordinate transformation
        def Phi_z(z):
            return self.K0 * z_transform(z)

        # Define twodeltaone with the phase term
        def twodeltaone(z):
            q = 1
            return (4 * np.pi / lambda_val) * calculate_refractive_index(lambda_val / 2, self.T) * z - \
                   (2 * (2 * np.pi / lambda_val) * calculate_refractive_index(lambda_val, self.T) * z +
                    q * Phi_z(z))

        # NCME solver
        nmce = NCME(self.kappa, twodeltaone)
        B_L = nmce.solve(self.L, self.A0, B0=0)

        # Calculate efficiency
        efficiency_val = efficiency(B_L, self.A0)

        return efficiency_val
