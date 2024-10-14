import numpy as np


def normalize(dx, dy, EX, EY, EZ, HX, HY, HZ):
    """
    Normalizes all of the field components so that the mode has
    unity power (Poynting vector integrated over the cross section.)

    Parameters:
    dx, dy: Grid spacings in x and y directions (can be scalar or vector)
    EX, EY, EZ: Electric field components
    HX, HY, HZ: Magnetic field components

    Returns:
    ex, ey, ez: Normalized electric field components
    hx, hy, hz: Normalized magnetic field components
    """
    Z0 = 119.9169832 * np.pi  # vacuum impedance
    nx, ny = EX.shape

    # Ensure dx and dy have the correct shape
    if np.isscalar(dx):
        dx = np.full(nx, dx)
    if np.isscalar(dy):
        dy = np.full(ny, dy)

    # Create indices for averaging
    ii = np.arange((nx + 1) * (ny + 1)).reshape(nx + 1, ny + 1)

    # Adjust indices to prevent out-of-bounds errors
    i1 = ii[0:nx, 1:ny + 1]  # Corresponds to the second column and beyond
    i2 = ii[0:nx, 0:ny]  # Corresponds to the first column
    i3 = ii[1:nx + 1, 0:ny]  # Rows shifted down by 1
    i4 = ii[1:nx + 1, 1:ny + 1]  # Both rows and columns shifted down by 1

    # Flatten the indices and unravel them for proper 2D indexing
    i1_rows, i1_cols = np.unravel_index(i1, HX.shape)
    i2_rows, i2_cols = np.unravel_index(i2, HX.shape)
    i3_rows, i3_cols = np.unravel_index(i3, HX.shape)
    i4_rows, i4_cols = np.unravel_index(i4, HX.shape)

    # Compute averaged magnetic field components
    HXp = (HX[i1_rows, i1_cols] + HX[i2_rows, i2_cols] +
           HX[i3_rows, i3_cols] + HX[i4_rows, i4_cols]) / 4
    HYp = (HY[i1_rows, i1_cols] + HY[i2_rows, i2_cols] +
           HY[i3_rows, i3_cols] + HY[i4_rows, i4_cols]) / 4

    # Compute the Poynting vector (Sz component)
    SZ = Z0 * (np.conj(EX) * HYp - np.conj(EY) * HXp +
               EX * np.conj(HYp) - EY * np.conj(HXp)) / 4

    # Compute the area elements
    dA = np.outer(dx, dy)

    # Normalize the fields
    N = np.sqrt(np.sum(SZ * dA))

    ex = Z0 * EX / N
    ey = Z0 * EY / N
    ez = Z0 * EZ / N
    hx = HX / N
    hy = HY / N
    hz = HZ / N

    return ex, ey, ez, hx, hy, hz
