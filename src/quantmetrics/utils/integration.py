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


def numeric_I_trap_array(u, theta, psi, levy_density, M=1.0, N=10_000, EXP_CLIP=700, chunk_u=None):
    """
    Compute I(u) = ∫_{-M}^{M} e^{theta*x + psi*x^2} (e^{i u x} - 1) nu(x) dx
    using trapezoid integration. Supports scalar or array u.

    Args:
        u (float or np.ndarray): Frequencies.
        theta (float): Linear distortion parameter.
        psi (float): Quadratic distortion parameter.
        levy_density (callable): Function nu(x), must accept vector x.
        M (float): Integration half-width.
        N (int): Number of trapezoid intervals.
        EXP_CLIP (float): Clip exponentials to avoid overflow.
        chunk_u (int or None): Optional chunk size for memory control.

    Returns:
        np.ndarray or complex: I(u) with same shape as u.
    """
    # Prepare x grid and static factors
    x = np.linspace(-M, M, N + 1)# Create a uniform grid x over [-M, M] with N+1 points
    h = x[1] - x[0] # spacing between grid points to be used in trapezoid integration.
    # compute g(x) = e^{theta x + psi x^2} * nu(x)
    expo_arg = theta * x + psi * x * x
    g = np.exp(np.clip(expo_arg, -EXP_CLIP, EXP_CLIP)) * levy_density(x)
    g = g.astype(np.complex128)

    # Trapezoid weights: 1 for interior, 0.5 for endpoints
    w = np.ones_like(x, dtype=np.complex128)
    w[0] = w[-1] = 0.5

    # Flatten u and prepare output. This converts u to a 1D complex array (even if scalar).
    u_arr = np.atleast_1d(u).astype(np.complex128)
    # Empty output array to store results.
    out = np.empty(u_arr.shape, dtype=np.complex128)

    # Sets a memory threshold: max number of complex elements allowed in phase matrix.
    # e.g., allow up to ~40 million complex elements (~640 MB)
    max_cells = 40_000_000 # Prevents allocating huge arrays that could crash the program.

    # Chunked integration subroutine
    def compute_chunk(u_chunk):
        # u_chunk is 1-D real array
        Nu, Nx = u_chunk.size, x.size # number of frequencies in this chunk and number of grid points
        if Nu * Nx <= max_cells: # if the full phase matrix fits in memory
            # full outer: shape (Nu, Nx)
            phase = np.exp(1j * np.outer(u_chunk, x)) - 1.0
            # weighted integrand and dot product over x
            # (phase * g) summed across x: yields shape (Nu,)
            return (phase * (g * w)).sum(axis=1) * h
        else: # if too large, print warning and iterate over u
            # print("Chunk size too large, iterating over u to limit memory.")
            # iterate over u in this chunk to limit memory
            res = np.empty(Nu, dtype=np.complex128)
            for j, uj in enumerate(u_chunk):
                phase = np.exp(1j * uj * x) - 1.0
                res[j] = h * np.dot(phase, g * w)
            return res

    if chunk_u is None: # Auto chunking
        # If total size fits in memory, do all at once
        if u_arr.size * x.size <= max_cells:
            out = compute_chunk(u_arr)
        else:
            # too large, do chunking
            # print("Input u array too large, auto-chunking...")
            # choose chunk size to keep Nu_chunk * Nx <= max_cells
            chunk_size = max(1, int(max_cells // x.size))
            for start in range(0, u_arr.size, chunk_size):
                # Loop over chunks of u and fill out
                stop = start + chunk_size
                out[start:stop] = compute_chunk(u_arr[start:stop])
    else:
        # user-specified chunking
        chunk_size = max(1, int(chunk_u))
        for start in range(0, u_arr.size, chunk_size):
            stop = start + chunk_size
            out[start:stop] = compute_chunk(u_arr[start:stop])

    # If input was scalar, return scalar
    return out.reshape(np.shape(u)) if np.ndim(u) > 0 else out.reshape(())