import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix, bmat


def neff_Hxy(lambda_, guess, nmodes, dx, dy, *args):
    '''
    This function computes the two transverse magnetic field
    components of a dielectric waveguide, using the finite
    difference method. For details about the method, please
    consult:  

    A. B. Fallahkhair, K. S. Li and T. E. Murphy, "Vector Finite
    Difference Modesolver for Anisotropic Dielectric
    Waveguides", J. Lightwave Technol. 26(11), 1423-1431,
    (2008). 

    USAGE:

    hx, hy, neff = wgmodes(lambda_, guess, nmodes, dx, dy, 
                           eps, boundary)
    hx, hy, neff = wgmodes(lambda_, guess, nmodes, dx, dy, 
                           epsxx, epsyy, epszz, boundary)
    hx, hy, neff = wgmodes(lambda_, guess, nmodes, dx, dy, 
                           epsxx, epsxy, epsyx, epsyy, epszz, boundary)

    INPUT:

    lambda_ - optical wavelength
    guess - scalar shift to apply when calculating the eigenvalues.
        This routine will return the eigenpairs which have an
        effective index closest to this guess
    nmodes - the number of modes to calculate
    dx - horizontal grid spacing (array or scalar)
    dy - vertical grid spacing (array or scalar)
    eps - index mesh (isotropic materials)  OR:
    epsxx, epsxy, epsyx, epsyy, epszz - index mesh (anisotropic)
    boundary - 4 letter string specifying boundary conditions to be
    applied at the edges of the computation window.  
        boundary[0] = North boundary condition
        boundary[1] = South boundary condition
        boundary[2] = East boundary condition
        boundary[3] = West boundary condition
    The following boundary conditions are supported: 
        'A' - Hx is antisymmetric, Hy is symmetric.
        'S' - Hx is symmetric and, Hy is antisymmetric.
        '0' - Hx and Hy are zero immediately outside of the
              boundary. 

    OUTPUT:

    hx - three-dimensional array containing Hx for each
         calculated mode 
    hy - three-dimensional array containing Hy for each
         calculated mode (e.g.: hy[:, :, k] = two-dimensional Hy
         matrix for the k-th mode)
    neff - array of modal effective indices

    NOTES:

    1) The units are arbitrary, but they must be self-consistent
    (e.g., if lambda is in um, then dx and dy should also be in
    um.)

    2) Unlike the E-field modesolvers, this method calculates
    the transverse MAGNETIC field components Hx and Hy.  Also,
    it calculates the components at the edges (vertices) of
    each cell, rather than in the center of each cell.  As a
    result, if size(eps) = [n, m], then the output eigenvectors
    will have a size of [n+1, m+1].

    3) This version of the modesolver can optionally support
    non-uniform grid sizes.  To use this feature, you may let dx
    and/or dy be arrays instead of scalars.

    4) The modesolver can consider anisotropic materials, provided
    the permittivity of all constituent materials can be
    expressed in one of the following forms:   

    [[eps,   0,    0],    [[epsxx,   0,     0],    [[epsxx, epsxy,   0],
     [  0,  eps,   0],     [  0,   epsyy,   0],     [epsyx, epsyy,   0],
     [  0,   0,  eps]]     [  0,     0,   epszz]]   [  0,     0,   epszz]]

    The program will decide which form is appropriate based upon
    the number of input arguments supplied.

    5) Perfectly matched boundary layers can be accommodated by
    using the complex coordinate stretching technique at the
    edges of the computation window.  (stretchmesh.py can be used
    for complex or real-coordinate stretching.)

    AUTHORS:  Thomas E. Murphy (tem@umd.edu)
              Arman B. Fallahkhair (a.b.fallah@gmail.com)
              Kai Sum Li (ksl3@njit.edu)
    '''
    nargin = len(args) + \
        5  # total number of input arguments including fixed ones

    if nargin == 11:
        epsxx = args[0]
        epsxy = args[1]
        epsyx = args[2]
        epsyy = args[3]
        epszz = args[4]
        boundary = args[5]
    elif nargin == 9:
        epsxx = args[0]
        epsxy = np.zeros_like(epsxx)
        epsyx = np.zeros_like(epsxx)
        epsyy = args[1]
        epszz = args[2]
        boundary = args[3]
    elif nargin == 7:
        epsxx = args[0]
        epsxy = np.zeros_like(epsxx)
        epsyx = np.zeros_like(epsxx)
        epsyy = epsxx
        epszz = epsxx
        boundary = args[1]
    else:
        raise ValueError('Incorrect number of input arguments.')

    nx, ny = epsxx.shape
    nx += 1
    ny += 1

    # Now we pad eps on all sides by one grid point
    epsxx = np.hstack((epsxx[:, [0]], epsxx, epsxx[:, [-1]]))
    epsxx = np.vstack((epsxx[[0], :], epsxx, epsxx[[-1], :]))
    epsyy = np.hstack((epsyy[:, [0]], epsyy, epsyy[:, [-1]]))
    epsyy = np.vstack((epsyy[[0], :], epsyy, epsyy[[-1], :]))
    epsxy = np.hstack((epsxy[:, [0]], epsxy, epsxy[:, [-1]]))
    epsxy = np.vstack((epsxy[[0], :], epsxy, epsxy[[-1], :]))
    epsyx = np.hstack((epsyx[:, [0]], epsyx, epsyx[:, [-1]]))
    epsyx = np.vstack((epsyx[[0], :], epsyx, epsyx[[-1], :]))
    epszz = np.hstack((epszz[:, [0]], epszz, epszz[:, [-1]]))
    epszz = np.vstack((epszz[[0], :], epszz, epszz[[-1], :]))

    k = 2 * np.pi / lambda_  # free-space wavevector

    if np.isscalar(dx):
        dx = np.asarray(dx, dtype=float) * np.ones(nx + 1)  # uniform grid
    else:
        dx = np.asarray(dx, dtype=float).flatten()
        dx = np.concatenate(([dx[0]], dx, [dx[-1]]))  # 両端にパディングを追加
    if np.isscalar(dy):
        dy = np.asarray(dy, dtype=float) * np.ones(ny + 1)  # uniform grid
    else:
        dy = np.asarray(dy).flatten()
        dy = np.concatenate(([dy[0]], dy, [dy[-1]]))  # pad dy on both ends

    # Distance to neighboring points to north, south, east, and west
    n = np.outer(np.ones(nx), np.asarray(dy[1:ny+1], dtype=float)).flatten()
    s = np.outer(np.ones(nx), np.asarray(dy[0:ny], dtype=float)).flatten()
    e = np.outer(np.asarray(dx[1:nx+1], dtype=float), np.ones(ny)).flatten()
    w = np.outer(np.asarray(dx[0:nx], dtype=float), np.ones(ny)).flatten()

    # epsilon tensor elements in regions 1,2,3,4, relative to the
    # mesh point under consideration (P), as shown below.
    #
    #                 NW------N------NE
    #                 |       |       |
    #                 |   1   n   4   |
    #                 |       |       |
    #                 W---w---P---e---E
    #                 |       |       |
    #                 |   2   s   3   |
    #                 |       |       |
    #                 SW------S------SE

    exx1 = np.ones(nx*ny)
    exx1[:] = epsxx[0:nx, 1:ny+1].flatten()
    exx2 = np.ones(nx*ny)
    exx2[:] = epsxx[0:nx, 0:ny].flatten()
    exx3 = np.ones(nx*ny)
    exx3[:] = epsxx[1:nx+1, 0:ny].flatten()
    exx4 = np.ones(nx*ny)
    exx4[:] = epsxx[1:nx+1, 1:ny+1].flatten()

    eyy1 = np.ones(nx*ny)
    eyy1[:] = epsyy[0:nx, 1:ny+1].flatten()
    eyy2 = np.ones(nx*ny)
    eyy2[:] = epsyy[0:nx, 0:ny].flatten()
    eyy3 = np.ones(nx*ny)
    eyy3[:] = epsyy[1:nx+1, 0:ny].flatten()
    eyy4 = np.ones(nx*ny)
    eyy4[:] = epsyy[1:nx+1, 1:ny+1].flatten()

    exy1 = np.ones(nx*ny)
    exy1[:] = epsxy[0:nx, 1:ny+1].flatten()
    exy2 = np.ones(nx*ny)
    exy2[:] = epsxy[0:nx, 0:ny].flatten()
    exy3 = np.ones(nx*ny)
    exy3[:] = epsxy[1:nx+1, 0:ny].flatten()
    exy4 = np.ones(nx*ny)
    exy4[:] = epsxy[1:nx+1, 1:ny+1].flatten()

    eyx1 = np.ones(nx*ny)
    eyx1[:] = epsyx[0:nx, 1:ny+1].flatten()
    eyx2 = np.ones(nx*ny)
    eyx2[:] = epsyx[0:nx, 0:ny].flatten()
    eyx3 = np.ones(nx*ny)
    eyx3[:] = epsyx[1:nx+1, 0:ny].flatten()
    eyx4 = np.ones(nx*ny)
    eyx4[:] = epsyx[1:nx+1, 1:ny+1].flatten()

    ezz1 = np.ones(nx*ny)
    ezz1[:] = epszz[0:nx, 1:ny+1].flatten()
    ezz2 = np.ones(nx*ny)
    ezz2[:] = epszz[0:nx, 0:ny].flatten()
    ezz3 = np.ones(nx*ny)
    ezz3[:] = epszz[1:nx+1, 0:ny].flatten()
    ezz4 = np.ones(nx*ny)
    ezz4[:] = epszz[1:nx+1, 1:ny+1].flatten()

    ns21 = n * eyy2 + s * eyy1
    ns34 = n * eyy3 + s * eyy4
    ew14 = e * exx1 + w * exx4
    ew23 = e * exx2 + w * exx3

    axxn = ((2 * eyy4 * e - eyx4 * n) * (eyy3 / ezz4) / ns34 +
            (2 * eyy1 * w + eyx1 * n) * (eyy2 / ezz1) / ns21) / (n * (e + w))

    axxs = ((2 * eyy3 * e + eyx3 * s) * (eyy4 / ezz3) / ns34 +
            (2 * eyy2 * w - eyx2 * s) * (eyy1 / ezz2) / ns21) / (s * (e + w))

    ayye = (2 * n * exx4 - e * exy4) * exx1 / ezz4 / e / ew14 / (n + s) + \
           (2 * s * exx3 + e * exy3) * exx2 / ezz3 / e / ew23 / (n + s)

    ayyw = (2 * exx1 * n + exy1 * w) * exx4 / ezz1 / w / ew14 / (n + s) + \
           (2 * exx2 * s - exy2 * w) * exx3 / ezz2 / w / ew23 / (n + s)

    axxe = 2 / (e * (e + w)) + \
        (eyy4 * eyx3 / ezz3 - eyy3 * eyx4 / ezz4) / (e + w) / ns34

    axxw = 2 / (w * (e + w)) + \
        (eyy2 * eyx1 / ezz1 - eyy1 * eyx2 / ezz2) / (e + w) / ns21

    ayyn = 2 / (n * (n + s)) + \
        (exx4 * exy1 / ezz1 - exx1 * exy4 / ezz4) / (n + s) / ew14

    ayys = 2 / (s * (n + s)) + \
        (exx2 * exy3 / ezz3 - exx3 * exy2 / ezz2) / (n + s) / ew23

    axxne = +eyx4 * eyy3 / ezz4 / (e + w) / ns34
    axxse = -eyx3 * eyy4 / ezz3 / (e + w) / ns34
    axxnw = -eyx1 * eyy2 / ezz1 / (e + w) / ns21
    axxsw = +eyx2 * eyy1 / ezz2 / (e + w) / ns21
    ayyne = +exy4 * exx1 / ezz4 / (n + s) / ew14
    ayyse = -exy3 * exx2 / ezz3 / (n + s) / ew23
    ayynw = -exy1 * exx4 / ezz1 / (n + s) / ew14
    ayysw = +exy2 * exx3 / ezz2 / (n + s) / ew23
    axxp = (- axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw
            + k**2 * (n + s) * (eyy4 * eyy3 * e / ns34 + eyy1 * eyy2 * w / ns21) / (e + w))
    ayyp = (- ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw
            + k**2 * (e + w) * (exx1 * exx4 * n / ew14 + exx2 * exx3 * s / ew23) / (n + s))
    axyn = (eyy3 * eyy4 / ezz4 / ns34 -
            eyy2 * eyy1 / ezz1 / ns21 +
            s * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
    axys = (eyy1 * eyy2 / ezz2 / ns21 -
            eyy4 * eyy3 / ezz3 / ns34 +
            n * (eyy2 * eyy4 - eyy1 * eyy3) / ns21 / ns34) / (e + w)
    ayxe = (exx1 * exx4 / ezz4 / ew14 -
            exx2 * exx3 / ezz3 / ew23 +
            w * (exx2 * exx4 - exx1 * exx3) / ew23 / ew14) / (n + s)
    ayxw = (exx3 * exx2 / ezz2 / ew23 -
            exx4 * exx1 / ezz1 / ew14 +
            e * (exx4 * exx2 - exx1 * exx3) / ew23 / ew14) / (n + s)

    axye = (eyy4 * (1 - eyy3 / ezz3) - eyy3 * (1 - eyy4 / ezz4)) / ns34 / (e + w) - \
        2 * (eyx1 * eyy2 / ezz1 * n * w / ns21 +
             eyx2 * eyy1 / ezz2 * s * w / ns21 +
             eyx4 * eyy3 / ezz4 * n * e / ns34 +
             eyx3 * eyy4 / ezz3 * s * e / ns34 +
             eyy1 * eyy2 * (1 / ezz1 - 1 / ezz2) * w**2 / ns21 +
             eyy3 * eyy4 * (1 / ezz4 - 1 / ezz3) * e * w / ns34) / e / (e + w)**2

    axyw = (eyy2 * (1 - eyy1 / ezz1) - eyy1 * (1 - eyy2 / ezz2)) / ns21 / (e + w) - \
        2 * (eyx4 * eyy3 / ezz4 * n * e / ns34 +
             eyx3 * eyy4 / ezz3 * s * e / ns34 +
             eyx1 * eyy2 / ezz1 * n * w / ns21 +
             eyx2 * eyy1 / ezz2 * s * w / ns21 +
             eyy4 * eyy3 * (1 / ezz3 - 1 / ezz4) * e**2 / ns34 +
             eyy2 * eyy1 * (1 / ezz2 - 1 / ezz1) * w * e / ns21) / w / (e + w)**2

    ayxn = (exx4 * (1 - exx1 / ezz1) - exx1 * (1 - exx4 / ezz4)) / ew14 / (n + s) - \
        2 * (exy3 * exx2 / ezz3 * e * s / ew23 +
             exy2 * exx3 / ezz2 * w * s / ew23 +
             exy4 * exx1 / ezz4 * e * n / ew14 +
             exy1 * exx4 / ezz1 * w * n / ew14 +
             exx3 * exx2 * (1 / ezz3 - 1 / ezz2) * s**2 / ew23 +
             exx1 * exx4 * (1 / ezz4 - 1 / ezz1) * n * s / ew14) / n / (n + s)**2

    ayxs = (exx2 * (1 - exx3 / ezz3) - exx3 * (1 - exx2 / ezz2)) / ew23 / (n + s) - \
        2 * (exy4 * exx1 / ezz4 * e * n / ew14 +
             exy1 * exx4 / ezz1 * w * n / ew14 +
             exy3 * exx2 / ezz3 * e * s / ew23 +
             exy2 * exx3 / ezz2 * w * s / ew23 +
             exx4 * exx1 * (1 / ezz1 - 1 / ezz4) * n**2 / ew14 +
             exx2 * exx3 * (1 / ezz2 - 1 / ezz3) * s * n / ew23) / s / (n + s)**2

    axyne = +eyy3 * (1 - eyy4 / ezz4) / (e + w) / ns34
    axyse = -eyy4 * (1 - eyy3 / ezz3) / (e + w) / ns34
    axynw = -eyy2 * (1 - eyy1 / ezz1) / (e + w) / ns21
    axysw = +eyy1 * (1 - eyy2 / ezz2) / (e + w) / ns21
    ayxne = +exx1 * (1 - exx4 / ezz4) / (n + s) / ew14
    ayxse = -exx2 * (1 - exx3 / ezz3) / (n + s) / ew23
    ayxnw = -exx4 * (1 - exx1 / ezz1) / (n + s) / ew14
    ayxsw = +exx3 * (1 - exx2 / ezz2) / (n + s) / ew23

    axyp = -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw) \
           - k**2 * (w * (n * eyx1 * eyy2 + s * eyx2 * eyy1) / ns21 +
                     e * (s * eyx3 * eyy4 + n * eyx4 * eyy3) / ns34) / (e + w)

    ayxp = -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw) \
           - k**2 * (n * (w * exy1 * exx4 + e * exy4 * exx1) / ew14 +
                     s * (w * exy2 * exx3 + e * exy3 * exx2) / ew23) / (n + s)

    ii = np.arange(nx * ny).reshape((nx, ny))

    # NORTH boundary
    ib = np.zeros(nx, dtype=int)
    ib[:] = ii[0:nx, ny-1]

    sign = 0
    if boundary[0] == 'S':
        sign = 1
    elif boundary[0] == 'A':
        sign = -1
    elif boundary[0] == '0':
        sign = 0
    else:
        raise ValueError(
            f'Unrecognized north boundary condition: {boundary[0]}')

    axxs[ib] += sign * axxn[ib]
    axxse[ib] += sign * axxne[ib]
    axxsw[ib] += sign * axxnw[ib]
    ayxs[ib] += sign * ayxn[ib]
    ayxse[ib] += sign * ayxne[ib]
    ayxsw[ib] += sign * ayxnw[ib]
    ayys[ib] -= sign * ayyn[ib]
    ayyse[ib] -= sign * ayyne[ib]
    ayysw[ib] -= sign * ayynw[ib]
    axys[ib] -= sign * axyn[ib]
    axyse[ib] -= sign * axyne[ib]
    axysw[ib] -= sign * axynw[ib]

    # SOUTH boundary
    ib = np.zeros(nx, dtype=int)
    ib[:] = ii[0:nx, 0]
    if boundary[1] == 'S':
        sign = +1
    elif boundary[1] == 'A':
        sign = -1
    elif boundary[1] == '0':
        sign = 0
    else:
        raise ValueError(
            f'Unrecognized south boundary condition: {boundary[1]}.')

    axxn[ib] += sign * axxs[ib]
    axxne[ib] += sign * axxse[ib]
    axxnw[ib] += sign * axxsw[ib]
    ayxn[ib] += sign * ayxs[ib]
    ayxne[ib] += sign * ayxse[ib]
    ayxnw[ib] += sign * ayxsw[ib]
    ayyn[ib] -= sign * ayys[ib]
    ayyne[ib] -= sign * ayyse[ib]
    ayynw[ib] -= sign * ayysw[ib]
    axyn[ib] -= sign * axys[ib]
    axyne[ib] -= sign * axyse[ib]
    axynw[ib] -= sign * axysw[ib]

    # EAST boundary
    ib = np.zeros(ny, dtype=int)
    ib[:] = ii[nx-1, 0:ny]
    if boundary[2] == 'S':
        sign = +1
    elif boundary[2] == 'A':
        sign = -1
    elif boundary[2] == '0':
        sign = 0
    else:
        raise ValueError(
            f'Unrecognized east boundary condition: {boundary[2]}.')

    axxw[ib] += sign * axxe[ib]
    axxnw[ib] += sign * axxne[ib]
    axxsw[ib] += sign * axxse[ib]
    ayxw[ib] += sign * ayxe[ib]
    ayxnw[ib] += sign * ayxne[ib]
    ayxsw[ib] += sign * ayxse[ib]
    ayyw[ib] -= sign * ayye[ib]
    ayynw[ib] -= sign * ayyne[ib]
    ayysw[ib] -= sign * ayyse[ib]
    axyw[ib] -= sign * axye[ib]
    axynw[ib] -= sign * axyne[ib]
    axysw[ib] -= sign * axyse[ib]

    # WEST boundary
    ib = np.zeros(ny, dtype=int)
    ib[:] = ii[0, 0:ny]
    if boundary[3] == 'S':
        sign = +1
    elif boundary[3] == 'A':
        sign = -1
    elif boundary[3] == '0':
        sign = 0
    else:
        raise ValueError(
            f'Unrecognized west boundary condition: {boundary[3]}.')

    axxe[ib] += sign * axxw[ib]
    axxne[ib] += sign * axxnw[ib]
    axxse[ib] += sign * axxsw[ib]
    ayxe[ib] += sign * ayxw[ib]
    ayxne[ib] += sign * ayxnw[ib]
    ayxse[ib] += sign * ayxsw[ib]
    ayye[ib] -= sign * ayyw[ib]
    ayyne[ib] -= sign * ayynw[ib]
    ayyse[ib] -= sign * ayysw[ib]
    axye[ib] -= sign * axyw[ib]
    axyne[ib] -= sign * axynw[ib]
    axyse[ib] -= sign * axysw[ib]

    # Assemble sparse matrix
    iall = ii.flatten()
    is_ = ii[0:nx, 0:ny-1].flatten()
    in_ = ii[0:nx, 1:ny].flatten()
    ie = ii[1:nx, 0:ny].flatten()
    iw = ii[0:nx-1, 0:ny].flatten()
    ine = ii[1:nx, 1:ny].flatten()
    ise = ii[1:nx, 0:ny-1].flatten()
    isw = ii[0:nx-1, 0:ny-1].flatten()
    inw = ii[0:nx-1, 1:ny].flatten()

    # Axx matrix
    Axx_rows = np.concatenate([iall, iw, ie, is_, in_, ine, ise, isw, inw])
    Axx_cols = np.concatenate([iall, ie, iw, in_, is_, isw, inw, ine, ise])
    Axx_data = np.concatenate([axxp[iall], axxe[iw], axxw[ie], axxn[is_], axxs[in_],
                               axxsw[ine], axxnw[ise], axxne[isw], axxse[inw]])
    Axx = coo_matrix((Axx_data, (Axx_rows, Axx_cols)), shape=(nx*ny, nx*ny))

    # Axy matrix
    Axy_rows = Axx_rows
    Axy_cols = Axx_cols
    Axy_data = np.concatenate([axyp[iall], axye[iw], axyw[ie], axyn[is_], axys[in_],
                               axysw[ine], axynw[ise], axyne[isw], axyse[inw]])
    Axy = coo_matrix((Axy_data, (Axy_rows, Axy_cols)), shape=(nx*ny, nx*ny))

    # Ayx matrix
    Ayx_rows = Axx_rows
    Ayx_cols = Axx_cols
    Ayx_data = np.concatenate([ayxp[iall], ayxe[iw], ayxw[ie], ayxn[is_], ayxs[in_],
                               ayxsw[ine], ayxnw[ise], ayxne[isw], ayxse[inw]])
    Ayx = coo_matrix((Ayx_data, (Ayx_rows, Ayx_cols)), shape=(nx*ny, nx*ny))

    # Ayy matrix
    Ayy_rows = Axx_rows
    Ayy_cols = Axx_cols
    Ayy_data = np.concatenate([ayyp[iall], ayye[iw], ayyw[ie], ayyn[is_], ayys[in_],
                               ayysw[ine], ayynw[ise], ayyne[isw], ayyse[inw]])
    Ayy = coo_matrix((Ayy_data, (Ayy_rows, Ayy_cols)), shape=(nx*ny, nx*ny))

    # Assemble the full matrix A
    A = bmat([[Axx, Axy], [Ayx, Ayy]], format='csr')

    # Compute shift
    shift = (guess * k) ** 2

    # Solve for eigenvalues and eigenvectors
    vals, vecs = eigs(  # type: ignore
        A, k=nmodes, sigma=shift,
        which='LM', tol=1e-8  # type: ignore
    )

    # Calculate effective indices
    neff = lambda_ * np.sqrt(vals.real) / (2 * np.pi)

    # Initialize mode fields
    phix = np.zeros((nx, ny, nmodes), dtype=complex)
    phiy = np.zeros((nx, ny, nmodes), dtype=complex)

    # Normalize modes
    temp = np.zeros((nx*ny, 2), dtype=complex)
    for kk in range(nmodes):
        temp[:, 0] = vecs[:nx*ny, kk]
        temp[:, 1] = vecs[nx*ny:, kk]
        mags = np.sqrt(np.sum(np.abs(temp)**2, axis=1))
        ii_max = np.argmax(mags)
        mag = mags[ii_max]
        if np.abs(temp[ii_max, 0]) > np.abs(temp[ii_max, 1]):
            jj = 0
        else:
            jj = 1
        mag = mag * temp[ii_max, jj] / np.abs(temp[ii_max, jj])
        temp = temp / mag
        phix[:, :, kk] = temp[:, 0].reshape((nx, ny))
        phiy[:, :, kk] = temp[:, 1].reshape((nx, ny))

    # Return the mode fields and effective indices
    return neff, phix, phiy
