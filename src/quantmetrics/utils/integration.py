import numpy as np
from numpy.polynomial.legendre import leggauss

def integrate_affine(f, a, b, N):
    """
    Gauss–Legendre quadrature with affine transform from [-1, 1] to [a, b].

    Args:
        f (callable): Function to integrate.
        a, b (float): Integration bounds.
        N (int): Number of quadrature points.

    Returns:
        float: Approximate integral of f over [a, b].
    """
    nodes, weights = leggauss(N)
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    x = xm + xr * nodes
    w = weights * xr
    return np.dot(w, f(x))

def integrate_split(f, L=1e-12, M=1.0, N_center=150, N_tails=100):
    """
    Split integration over [-M, -L], [-L, L], [L, M] using Gauss–Legendre quadrature.

    Args:
        f (callable): Function to integrate.
        L, M (float): Inner and outer bounds.
        N_center, N_tails (int): Number of quadrature points.

    Returns:
        float: Approximate integral over [-M, M].
    """
    return (
        integrate_affine(f, -M, -L, N_tails) +
        integrate_affine(f, -L, L, N_center) +
        integrate_affine(f, L, M, N_tails)
    )