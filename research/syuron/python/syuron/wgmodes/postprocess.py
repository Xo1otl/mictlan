import numpy as np
from scipy.sparse import csr_matrix, hstack


def postprocess(lambda_, neff, Hx, Hy, dx, dy, *args):
    nargin = 6 + len(args)
    if nargin == 12:
        epsxx = args[0]
        epsxy = args[1]
        epsyx = args[2]
        epsyy = args[3]
        epszz = args[4]
        boundary = args[5]
    elif nargin == 10:
        epsxx = args[0]
        epsxy = np.zeros_like(epsxx)
        epsyx = np.zeros_like(epsxx)
        epsyy = args[1]
        epszz = args[2]
        boundary = args[3]
    elif nargin == 8:
        epsxx = args[0]
        epsxy = np.zeros_like(epsxx)
        epsyx = np.zeros_like(epsxx)
        epsyy = epsxx
        epszz = epsxx
        boundary = args[1]
    else:
        raise ValueError('Incorrect number of input arguments.')

    nx, ny = epsxx.shape
    nx = nx + 1
    ny = ny + 1

    # now we pad eps on all sides by one grid point
    def pad_eps(eps):
        eps = np.hstack((eps[:, [0]], eps, eps[:, [-1]]))
        eps = np.vstack((eps[[0], :], eps, eps[[-1], :]))
        return eps

    epsxx = pad_eps(epsxx)
    epsyy = pad_eps(epsyy)
    epsxy = pad_eps(epsxy)
    epsyx = pad_eps(epsyx)
    epszz = pad_eps(epszz)

    k = 2 * np.pi / lambda_  # free-space wavevector
    b = neff * k             # propagation constant (eigenvalue)

    if np.isscalar(dx):
        dx = np.array([dx] * (nx + 1))  # uniform grid
    else:
        dx = np.array(dx).flatten()
        dx = np.concatenate(([dx[0]], dx, [dx[-1]]))

    if np.isscalar(dy):
        dy = np.array([dy] * (ny + 1))  # uniform grid
    else:
        dy = np.array(dy).flatten()
        dy = np.concatenate(([dy[0]], dy, [dy[-1]]))

    # distance to neighboring points to north south east and west,
    # relative to point under consideration (P)
    n = np.ones((nx, ny)) * dy[1:ny + 1]
    s = np.ones((nx, ny)) * dy[0:ny]
    e = dx[1:nx + 1].reshape(-1, 1) * np.ones((1, ny))
    w = dx[0:nx].reshape(-1, 1) * np.ones((1, ny))

    n = n.flatten()
    s = s.flatten()
    e = e.flatten()
    w = w.flatten()

    # epsilon tensor elements in regions 1,2,3,4
    exx1 = epsxx[0:nx, 1:ny + 1].flatten()
    exx2 = epsxx[0:nx, 0:ny].flatten()
    exx3 = epsxx[1:nx + 1, 0:ny].flatten()
    exx4 = epsxx[1:nx + 1, 1:ny + 1].flatten()

    eyy1 = epsyy[0:nx, 1:ny + 1].flatten()
    eyy2 = epsyy[0:nx, 0:ny].flatten()
    eyy3 = epsyy[1:nx + 1, 0:ny].flatten()
    eyy4 = epsyy[1:nx + 1, 1:ny + 1].flatten()

    exy1 = epsxy[0:nx, 1:ny + 1].flatten()
    exy2 = epsxy[0:nx, 0:ny].flatten()
    exy3 = epsxy[1:nx + 1, 0:ny].flatten()
    exy4 = epsxy[1:nx + 1, 1:ny + 1].flatten()

    eyx1 = epsyx[0:nx, 1:ny + 1].flatten()
    eyx2 = epsyx[0:nx, 0:ny].flatten()
    eyx3 = epsyx[1:nx + 1, 0:ny].flatten()
    eyx4 = epsyx[1:nx + 1, 1:ny + 1].flatten()

    ezz1 = epszz[0:nx, 1:ny + 1].flatten()
    ezz2 = epszz[0:nx, 0:ny].flatten()
    ezz3 = epszz[1:nx + 1, 0:ny].flatten()
    ezz4 = epszz[1:nx + 1, 1:ny + 1].flatten()

    num_elements = nx * ny

    # Initialize all bzx and bzy variables with empty arrays
    bzxne = (1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx4 / ezz4 /
             (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
             eyy3 * eyy1 * w * eyy2 +
             1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (1 - exx4 / ezz4) / ezz3 / ezz2 / (w * exx3 + e * exx2) /
             (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b
    bzxse = (-1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * eyx3 / ezz3 /
             (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
             eyy4 * eyy1 * w * eyy2 +
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (1 - exx3 / ezz3) / (w * exx3 + e * exx2) / ezz4 / ezz1 /
             (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b
    bzxnw = (-1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx1 / ezz4 / ezz3 /
             (n * eyy3 + s * eyy4) / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
             eyy4 * eyy3 * eyy2 * e -
             1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (1 - exx1 / ezz1) / ezz3 / ezz2 /
             (w * exx3 + e * exx2) / (w * exx4 + e * exx1) /
             (n + s) * exx2 * exx3 * exx4 * s) / b
    bzxsw = (1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * eyx2 / ezz4 / ezz3 /
             (n * eyy3 + s * eyy4) / ezz2 / (n * eyy2 + s * eyy1) / (e + w) *
             eyy4 * eyy3 * eyy1 * e -
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (1 - exx2 / ezz2) / (w * exx3 + e * exx2) / ezz4 / ezz1 /
             (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b

    bzxn = ((1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * n * ezz1 * ezz2 / eyy1 *
             (2 * eyy1 / ezz1 / n**2 + eyx1 / ezz1 / n / w) +
             1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * n * ezz4 * ezz3 / eyy4 *
             (2 * eyy4 / ezz4 / n**2 - eyx4 / ezz4 / n / e)) / ezz4 / ezz3 /
            (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
            eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (1 / 2 * ezz4 * ((1 - exx1 / ezz1) / n / w - exy1 / ezz1 * (2 / n**2 - 2 / n**2 * s / (n + s))) /
              exx1 * ezz1 * w + (ezz4 - ezz1) * s / n / (n + s) +
              1 / 2 * ezz1 * (-(1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (2 / n**2 - 2 / n**2 * s / (n + s))) /
              exx4 * ezz4 * e) -
             (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (-ezz3 * exy2 / n / (n + s) / exx2 * w + (ezz3 - ezz2) * s / n / (n + s) -
              ezz2 * exy3 / n / (n + s) / exx3 * e)) / ezz3 / ezz2 /
            (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) /
            (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzxs = ((1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) * s * ezz2 * ezz1 / eyy2 *
             (2 * eyy2 / ezz2 / s**2 - eyx2 / ezz2 / s / w) +
             1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) * s * ezz3 * ezz4 / eyy3 *
             (2 * eyy3 / ezz3 / s**2 + eyx3 / ezz3 / s / e)) / ezz4 / ezz3 /
            (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
            eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (-ezz4 * exy1 / s / (n + s) / exx1 * w - (ezz4 - ezz1) * n / s / (n + s) -
              ezz1 * exy4 / s / (n + s) / exx4 * e) -
             (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (1 / 2 * ezz3 * (-(1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (2 / s**2 - 2 / s**2 * n / (n + s))) /
              exx2 * ezz2 * w - (ezz3 - ezz2) * n / s / (n + s) +
              1 / 2 * ezz2 * ((1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (2 / s**2 - 2 / s**2 * n / (n + s))) /
              exx3 * ezz3 * e)) / ezz3 / ezz2 /
            (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) /
            (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzxe = ((n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
            (1 / 2 * n * ezz4 * ezz3 / eyy4 * (2 / e**2 - eyx4 / ezz4 / n / e) +
             1 / 2 * s * ezz3 * ezz4 / eyy3 * (2 / e**2 + eyx3 / ezz3 / s / e)) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) *
            eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            (-1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             ezz1 * (1 - exx4 / ezz4) / n / exx4 * ezz4 -
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             ezz2 * (1 - exx3 / ezz3) / s / exx3 * ezz3) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) /
            ezz4 / ezz1 / (w * exx4 + e * exx1) /
            (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzxw = ((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
            (1 / 2 * n * ezz1 * ezz2 / eyy1 * (2 / w**2 + eyx1 / ezz1 / n / w) +
             1 / 2 * s * ezz2 * ezz1 / eyy2 * (2 / w**2 - eyx2 / ezz2 / s / w)) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) *
            eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            (1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             ezz4 * (1 - exx1 / ezz1) / n / exx1 * ezz1 +
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             ezz3 * (1 - exx2 / ezz2) / s / exx2 * ezz2) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) /
            ezz4 / ezz1 / (w * exx4 + e * exx1) /
            (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzxp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (1 / 2 * n * ezz1 * ezz2 / eyy1 * (-2 / w**2 - 2 * eyy1 / ezz1 / n**2 + k**2 * eyy1 - eyx1 / ezz1 / n / w) +
              1 / 2 * s * ezz2 * ezz1 / eyy2 * (-2 / w**2 - 2 * eyy2 / ezz2 / s**2 + k**2 * eyy2 + eyx2 / ezz2 / s / w)) +
             (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (1 / 2 * n * ezz4 * ezz3 / eyy4 * (-2 / e**2 - 2 * eyy4 / ezz4 / n**2 + k**2 * eyy4 + eyx4 / ezz4 / n / e) +
              1 / 2 * s * ezz3 * ezz4 / eyy3 * (-2 / e**2 - 2 * eyy3 / ezz3 / s**2 + k**2 * eyy3 - eyx3 / ezz3 / s / e))) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 / (n * eyy2 + s * eyy1) / (e + w) *
            eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (1 / 2 * ezz4 * (-k**2 * exy1 - (1 - exx1 / ezz1) / n / w - exy1 / ezz1 * (-2 / n**2 - 2 / n**2 * (n - s) / s)) /
              exx1 * ezz1 * w + (ezz4 - ezz1) * (n - s) / n / s +
              1 / 2 * ezz1 * (-k**2 * exy4 + (1 - exx4 / ezz4) / n / e - exy4 / ezz4 * (-2 / n**2 - 2 / n**2 * (n - s) / s)) /
              exx4 * ezz4) -
             (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (1 / 2 * ezz3 * (-k**2 * exy2 + (1 - exx2 / ezz2) / s / w - exy2 / ezz2 * (-2 / s**2 + 2 / s**2 * (n - s) / n)) /
              exx2 * ezz2 * w + (ezz3 - ezz2) * (n - s) / n / s +
              1 / 2 * ezz2 * (-k**2 * exy3 - (1 - exx3 / ezz3) / s / e - exy3 / ezz3 * (-2 / s**2 + 2 / s**2 * (n - s) / n)) /
              exx3 * ezz3)) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 / (w * exx4 + e * exx1) /
            (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzyne = (1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (1 - eyy4 / ezz4) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
             (n * eyy2 + s * eyy1) / (e + w) * eyy3 * eyy1 * w * eyy2 +
             1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             exy4 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 /
             (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx1 * s) / b
    bzyse = (-1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (1 - eyy3 / ezz3) / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
             (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy1 * w * eyy2 +
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             exy3 / ezz3 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
             (w * exx4 + e * exx1) / (n + s) * exx2 * n * exx1 * exx4) / b
    bzynw = (-1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (1 - eyy1 / ezz1) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) /
             (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy2 * e -
             1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             exy1 / ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz1 /
             (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * exx4 * s) / b
    bzysw = (1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (1 - eyy2 / ezz2) / ezz4 / ezz3 / (n * eyy3 + s * eyy4) /
             (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * e -
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             exy2 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
             (w * exx4 + e * exx1) / (n + s) * exx3 * n * exx1 * exx4) / b
    bzyn = ((1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             ezz1 * ezz2 / eyy1 * (1 - eyy1 / ezz1) / w -
             1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             ezz4 * ezz3 / eyy4 * (1 - eyy4 / ezz4) / e) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
            (1 / 2 * ezz4 * (2 / n**2 + exy1 / ezz1 / n / w) / exx1 * ezz1 * w +
             1 / 2 * ezz1 * (2 / n**2 - exy4 / ezz4 / n / e) / exx4 * ezz4 * e) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
            (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzys = ((-1 / 2 * (-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             ezz2 * ezz1 / eyy2 * (1 - eyy2 / ezz2) / w +
             1 / 2 * (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             ezz3 * ezz4 / eyy3 * (1 - eyy3 / ezz3) / e) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e -
            (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
            (1 / 2 * ezz3 * (2 / s**2 - exy2 / ezz2 / s / w) / exx2 * ezz2 * w +
             1 / 2 * ezz2 * (2 / s**2 + exy3 / ezz3 / s / e) / exx3 * ezz3 * e) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
            (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzye = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (-n * ezz2 / eyy1 * eyx1 / (e + w) + (ezz1 - ezz2) * w / e / (e + w) -
              s * ezz1 / eyy2 * eyx2 / e / (e + w)) +
             (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (1 / 2 * n * ezz4 * ezz3 / eyy4 * (-(1 - eyy4 / ezz4) / n / e -
                                                eyx4 / ezz4 * (2 / e**2 - 2 / e**2 * w / (e + w))) +
              1 / 2 * s * ezz3 * ezz4 / eyy3 * ((1 - eyy3 / ezz3) / s / e -
                                                eyx3 / ezz3 * (2 / e**2 - 2 / e**2 * w / (e + w))) +
              (ezz4 - ezz3) * w / e / (e + w))) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            (1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             ezz1 * (2 * exx4 / ezz4 / e**2 - exy4 / ezz4 / n / e) /
             exx4 * ezz4 * e -
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             ezz2 * (2 * exx3 / ezz3 / e**2 + exy3 / ezz3 / s / e) /
             exx3 * ezz3 * e) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
            (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzyw = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (1 / 2 * n * ezz1 * ezz2 / eyy1 * ((1 - eyy1 / ezz1) / n / w -
                                                eyx1 / ezz1 * (2 / w**2 - 2 / w**2 * e / (e + w))) -
              (ezz1 - ezz2) * e / w / (e + w) +
              1 / 2 * s * ezz2 * ezz1 / eyy2 * (-(1 - eyy2 / ezz2) / s / w -
                                                eyx2 / ezz2 * (2 / w**2 - 2 / w**2 * e / (e + w)))) +
             (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (-n * ezz3 / eyy4 * eyx4 / w / (e + w) -
              s * ezz4 / eyy3 * eyx3 / w / (e + w) -
              (ezz4 - ezz3) * e / w / (e + w))) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            (1 / 2 * (ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             ezz4 * (2 * exx1 / ezz1 / w**2 + exy1 / ezz1 / n / w) /
             exx1 * ezz1 * w -
             1 / 2 * (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             ezz3 * (2 * exx2 / ezz2 / w**2 - exy2 / ezz2 / s / w) /
             exx2 * ezz2 * w) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
            (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b
    bzyp = (((-n * ezz4 * ezz3 / eyy4 - s * ezz3 * ezz4 / eyy3) *
             (1 / 2 * n * ezz1 * ezz2 / eyy1 * (-k**2 * eyx1 - (1 - eyy1 / ezz1) / n / w -
                                                eyx1 / ezz1 * (-2 / w**2 + 2 / w**2 * (e - w) / e)) +
              (ezz1 - ezz2) * (e - w) / e / w +
              1 / 2 * s * ezz2 * ezz1 / eyy2 * (-k**2 * eyx2 + (1 - eyy2 / ezz2) / s / w -
                                                eyx2 / ezz2 * (-2 / w**2 + 2 / w**2 * (e - w) / e))) +
             (n * ezz1 * ezz2 / eyy1 + s * ezz2 * ezz1 / eyy2) *
             (1 / 2 * n * ezz4 * ezz3 / eyy4 * (-k**2 * eyx4 + (1 - eyy4 / ezz4) / n / e -
                                                eyx4 / ezz4 * (-2 / e**2 - 2 / e**2 * (e - w) / w)) +
              1 / 2 * s * ezz3 * ezz4 / eyy3 * (-k**2 * eyx3 - (1 - eyy3 / ezz3) / s / e -
                                                eyx3 / ezz3 * (-2 / e**2 - 2 / e**2 * (e - w) / w)) +
              (ezz4 - ezz3) * (e - w) / e / w)) /
            ezz4 / ezz3 / (n * eyy3 + s * eyy4) / ezz2 / ezz1 /
            (n * eyy2 + s * eyy1) / (e + w) * eyy4 * eyy3 * eyy1 * w * eyy2 * e +
            ((ezz3 / exx2 * ezz2 * w + ezz2 / exx3 * ezz3 * e) *
             (1 / 2 * ezz4 * (-2 / n**2 - 2 * exx1 / ezz1 / w**2 + k**2 * exx1 -
                              exy1 / ezz1 / n / w) / exx1 * ezz1 * w +
                1 / 2 * ezz1 * (-2 / n**2 - 2 * exx4 / ezz4 / e**2 + k**2 * exx4 +
                                exy4 / ezz4 / n / e) / exx4 * ezz4 * e) -
             (ezz4 / exx1 * ezz1 * w + ezz1 / exx4 * ezz4 * e) *
             (1 / 2 * ezz3 * (-2 / s**2 - 2 * exx2 / ezz2 / w**2 + k**2 * exx2 +
                              exy2 / ezz2 / s / w) / exx2 * ezz2 * w +
                1 / 2 * ezz2 * (-2 / s**2 - 2 * exx3 / ezz3 / e**2 + k**2 * exx3 -
                                exy3 / ezz3 / s / e) / exx3 * ezz3 * e)) /
            ezz3 / ezz2 / (w * exx3 + e * exx2) / ezz4 / ezz1 /
            (w * exx4 + e * exx1) / (n + s) * exx2 * exx3 * n * exx1 * exx4 * s) / b

    # Initialize index arrays
    ii = np.arange(nx * ny).reshape(nx, ny)

    # NORTH boundary
    ib = ii[:, ny - 1]
    if boundary[0] == 'S':
        sign = +1
    elif boundary[0] == 'A':
        sign = -1
    elif boundary[0] == '0':
        sign = 0
    else:
        raise ValueError(
            'Unrecognized north boundary condition: %s.' % boundary[0])
    bzxs[ib] = bzxs[ib] + sign * bzxn[ib]
    bzxse[ib] = bzxse[ib] + sign * bzxne[ib]
    bzxsw[ib] = bzxsw[ib] + sign * bzxnw[ib]
    bzys[ib] = bzys[ib] - sign * bzyn[ib]
    bzyse[ib] = bzyse[ib] - sign * bzyne[ib]
    bzysw[ib] = bzysw[ib] - sign * bzynw[ib]

    # SOUTH boundary
    ib = ii[:, 0]
    if boundary[1] == 'S':
        sign = +1
    elif boundary[1] == 'A':
        sign = -1
    elif boundary[1] == '0':
        sign = 0
    else:
        raise ValueError(
            'Unrecognized south boundary condition: %s.' % boundary[1])
    bzxn[ib] = bzxn[ib] + sign * bzxs[ib]
    bzxne[ib] = bzxne[ib] + sign * bzxse[ib]
    bzxnw[ib] = bzxnw[ib] + sign * bzxsw[ib]
    bzyn[ib] = bzyn[ib] - sign * bzys[ib]
    bzyne[ib] = bzyne[ib] - sign * bzyse[ib]
    bzynw[ib] = bzynw[ib] - sign * bzysw[ib]

    # EAST boundary
    ib = ii[nx - 1, :]
    if boundary[2] == 'S':
        sign = +1
    elif boundary[2] == 'A':
        sign = -1
    elif boundary[2] == '0':
        sign = 0
    else:
        raise ValueError(
            'Unrecognized east boundary condition: %s.' % boundary[2])
    bzxw[ib] = bzxw[ib] + sign * bzxe[ib]
    bzxnw[ib] = bzxnw[ib] + sign * bzxne[ib]
    bzxsw[ib] = bzxsw[ib] + sign * bzxse[ib]
    bzyw[ib] = bzyw[ib] - sign * bzye[ib]
    bzynw[ib] = bzynw[ib] - sign * bzyne[ib]
    bzysw[ib] = bzysw[ib] - sign * bzyse[ib]

    # WEST boundary
    ib = ii[0, :]
    if boundary[3] == 'S':
        sign = +1
    elif boundary[3] == 'A':
        sign = -1
    elif boundary[3] == '0':
        sign = 0
    else:
        raise ValueError(
            'Unrecognized west boundary condition: %s.' % boundary[3])
    bzxe[ib] = bzxe[ib] + sign * bzxw[ib]
    bzxne[ib] = bzxne[ib] + sign * bzxnw[ib]
    bzxse[ib] = bzxse[ib] + sign * bzxsw[ib]
    bzye[ib] = bzye[ib] - sign * bzyw[ib]
    bzyne[ib] = bzyne[ib] - sign * bzynw[ib]
    bzyse[ib] = bzyse[ib] - sign * bzysw[ib]

    # Assemble sparse matrices
    iall = ii.flatten()
    is_idx = ii[:, 0:ny - 1].flatten()
    in_idx = ii[:, 1:ny].flatten()
    ie_idx = ii[1:nx, :].flatten()
    iw_idx = ii[0:nx - 1, :].flatten()
    ine_idx = ii[1:nx, 1:ny].flatten()
    ise_idx = ii[1:nx, 0:ny - 1].flatten()
    isw_idx = ii[0:nx - 1, 0:ny - 1].flatten()
    inw_idx = ii[0:nx - 1, 1:ny].flatten()

    # Assemble the sparse matrices Bzx and Bzy
    row_indices = np.concatenate([
        iall, iw_idx, ie_idx, is_idx, in_idx, ine_idx, ise_idx, isw_idx, inw_idx])
    col_indices = np.concatenate([
        iall, ie_idx, iw_idx, in_idx, is_idx, isw_idx, inw_idx, ine_idx, ise_idx])
    data_bzx = np.concatenate([
        bzxp[iall], bzxe[iw_idx], bzxw[ie_idx], bzxn[is_idx], bzxs[in_idx],
        bzxsw[ine_idx], bzxnw[ise_idx], bzxne[isw_idx], bzxse[inw_idx]])
    data_bzy = np.concatenate([
        bzyp[iall], bzye[iw_idx], bzyw[ie_idx], bzyn[is_idx], bzys[in_idx],
        bzysw[ine_idx], bzynw[ise_idx], bzyne[isw_idx], bzyse[inw_idx]])

    Bzx = csr_matrix((data_bzx, (row_indices, col_indices)),
                     shape=(nx * ny, nx * ny))
    Bzy = csr_matrix((data_bzy, (row_indices, col_indices)),
                     shape=(nx * ny, nx * ny))

    B = hstack([Bzx, Bzy])

    # Compute Hz
    Hz = np.zeros_like(Hx, dtype=complex)
    H_combined = np.concatenate([Hx.flatten(), Hy.flatten()])
    Hz_flat = B.dot(H_combined) / (1j)
    Hz = Hz_flat.reshape(Hx.shape)

    # Adjust nx and ny for electric field computations
    nx = nx - 1
    ny = ny - 1

    exx = epsxx[1:nx + 1, 1:ny + 1]
    exy = epsxy[1:nx + 1, 1:ny + 1]
    eyx = epsyx[1:nx + 1, 1:ny + 1]
    eyy = epsyy[1:nx + 1, 1:ny + 1]
    ezz = epszz[1:nx + 1, 1:ny + 1]
    edet = (exx * eyy - exy * eyx)

    h = dx[1:nx + 1].reshape(-1, 1) * np.ones((1, ny))
    v = np.ones((nx, 1)) * dy[1:ny + 1]

    i1 = ii[0:nx, 1:ny + 1]
    i2 = ii[0:nx, 0:ny]
    i3 = ii[1:nx + 1, 0:ny]
    i4 = ii[1:nx + 1, 1:ny + 1]

    # フラットなインデックスを2Dインデックスに変換
    i1_rows, i1_cols = np.unravel_index(i1, Hz.shape)
    i2_rows, i2_cols = np.unravel_index(i2, Hz.shape)
    i3_rows, i3_cols = np.unravel_index(i3, Hz.shape)
    i4_rows, i4_cols = np.unravel_index(i4, Hz.shape)

    print(Hy.shape)
    print(i1.shape, i2.shape, i3.shape, i4.shape)
    print(Hy[i1_rows, i1_cols])

    # Dx = (+neff * (Hy[i1] + Hy[i2] + Hy[i3] + Hy[i4]) / 4 +
    #       (Hz[i1] + Hz[i4] - Hz[i2] - Hz[i3]) / (1j * 2 * k * v))
    # Dy = (-neff * (Hx[i1] + Hx[i2] + Hx[i3] + Hx[i4]) / 4 -
    #       (Hz[i3] + Hz[i4] - Hz[i1] - Hz[i2]) / (1j * 2 * k * h))
    # Dz = ((Hy[i3] + Hy[i4] - Hy[i1] - Hy[i2]) / (2 * h) -
    #       (Hx[i1] + Hx[i4] - Hx[i2] - Hx[i3]) / (2 * v)) / (1j * k)

    # Ex = (eyy * Dx - exy * Dy) / edet
    # Ey = (exx * Dy - eyx * Dx) / edet
    # Ez = Dz / ezz

    # return Hz, Ex, Ey, Ez
    
    # 各フィールドの計算
    Dx = (+neff * (Hy[i1_rows, i1_cols] + Hy[i2_rows, i2_cols] + Hy[i3_rows, i3_cols] + Hy[i4_rows, i4_cols]) / 4 +
          (Hz[i1_rows, i1_cols] + Hz[i4_rows, i4_cols] - Hz[i2_rows, i2_cols] - Hz[i3_rows, i3_cols]) / (1j * 2 * k * v))
    
    Dy = (-neff * (Hx[i1_rows, i1_cols] + Hx[i2_rows, i2_cols] + Hx[i3_rows, i3_cols] + Hx[i4_rows, i4_cols]) / 4 -
          (Hz[i3_rows, i3_cols] + Hz[i4_rows, i4_cols] - Hz[i1_rows, i1_cols] - Hz[i2_rows, i2_cols]) / (1j * 2 * k * h))
    
    Dz = ((Hy[i3_rows, i3_cols] + Hy[i4_rows, i4_cols] - Hy[i1_rows, i1_cols] - Hy[i2_rows, i2_cols]) / (2 * h) -
          (Hx[i1_rows, i1_cols] + Hx[i4_rows, i4_cols] - Hx[i2_rows, i2_cols] - Hx[i3_rows, i3_cols]) / (2 * v)) / (1j * k)
    
    # 電場の計算
    Ex = (eyy * Dx - exy * Dy) / edet
    Ey = (exx * Dy - eyx * Dx) / edet
    Ez = Dz / ezz
    
    # 結果を返す
    return Hz, Ex, Ey, Ez
    