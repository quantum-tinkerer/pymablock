from perturbative_model import PerturbativeModel
import numpy as np
import sympy

one = sympy.sympify(1)


def _perturbative_kpm_vector_generator(ham, vectors, max_moments):
    """
    Generator object that succesively yields KPM vectors `T_n(ham) |vector>`
    for vectors in `vectors` for n in [0, max_moments].

    Parameters
    ----------
    ham : PerturbativeModel
        Hamiltonian, PerturbativeModel with shape `(N, N)`.
        The `interesting_keys` property is used to limit the powers of
        free parameters that we keep track of in the expansion.
    vectors : 1D or 2D array or PerturbativeModel
        Vector of length `N` or array of vectors with shape `(M, N)`.
        If PerturbativeModel, must be of shape `(M, N)`. The size of
        the last index should be the same as the size of `ham` `N`.
    max_moments : int
        Number of moments to stop with iteration

    Notes
    -----
    Returns a sequence of expanded vectors, PerturbativeModels of shape (M, N).
    If the input was a vector, M=1.
    """
    if not isinstance(vectors, PerturbativeModel):
        alpha = PerturbativeModel({one: np.atleast_2d(vectors)})
    # Internally store as column vectors
    alpha = alpha.T()
    n = 0
    yield alpha.T()
    n += 1
    alpha_prev = alpha
    alpha = ham * alpha
    yield alpha.T()
    n += 1
    while n < max_moments:
        alpha_next = 2 * ham * alpha - alpha_prev
        alpha_prev = alpha
        alpha = alpha_next
        yield alpha.T()
        n += 1
