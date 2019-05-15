import numpy as np
import sympy
import scipy.sparse

import kwant.kpm
from kwant._common import ensure_rng

from .perturbative_model import PerturbativeModel
from .higher_order_lowdin import get_interesting_keys

one = sympy.sympify(1)


def trace_perturbation(H0, H1, order=2, interesting_keys=None,
                       trace=True, kpm_params=dict()):
    """
    Perturbative expansion of `Tr(f(H) * A)` using stochastic trace.

    Parameters
    ----------

    H0 : array or PerturbativeModel
        Unperturbed hamiltonian, dense or sparse matrix. If
        provided as PerturbativeModel, it must contain a single
        entry {1 : array}.
    H1 : dict of {symbol : array} or PerturbativeModel
        Perturbation to the Hamiltonian.
    order : int (default 2)
        Order of the perturbation calculation.
    interesting_keys : iterable of sympy expressions or None (default)
        By default up to `order` order polynomials of every key in `H1`
        is kept. If not all of these are interesting, the calculation
        can be sped up by providing a subset of these keys to keep.
        Should be a subset of the keys up to `order` and should contain
        all subexpressions of desired keys, as terms not listed in
        `interesting_keys` are discarded at every step of the calculation.
    trace : bool, default True
        Whether to calculate the trace. If False, all matrix elements
        between pairs of vectors are separately returned.
    kpm_params : dict, optional
        Dictionary containing the parameters, see `~kwant.kpm`.

        num_vectors : int, default 10
            Number of random vectors used in KPM. Ignored if `vector_factory`
            is provided.
        num_moments : int, default 100
            Number of moments in the KPM expansion.
        operator : 2D array or PerturbativeModel or None (default)
            Operator in the expectation value, default is identity.
        vector_factory : 1D or 2D array or PerturbativeModel or None (default)
            Vector of length `N` or array of vectors with shape `(M, N)`.
            If PerturbativeModel, must be of shape `(M, N)`. The size of
            the last index should be the same as the size of the Hamiltonian.
            By default, random vectors are used.
    """
    H1 = PerturbativeModel(H1)
    H1 = H1.tosparse()
    all_keys = get_interesting_keys(H1.keys(), order)
    if interesting_keys is None:
        interesting_keys = all_keys
    else:
        interesting_keys = set(interesting_keys)
    H1.interesting_keys = interesting_keys
    if not interesting_keys <= all_keys:
        raise ValueError('`interesting_keys` should be a subset of all monomials of `H1.keys()` '
                         'up to total power `order`.')

    # Convert to appropriate format
    if not isinstance(H0, PerturbativeModel):
        H0 = PerturbativeModel({one: H0}, interesting_keys=interesting_keys)
    elif not (len(H0) == 1 and list(H0.keys()).pop() == one):
        raise ValueError('H0 must contain a single entry {sympy.sympify(1): array}.')
    H0 = H0.tosparse()
    # Find the bounds of the spectrum and rescale `ham`
    eps = kpm_params.get('eps', 0.05)
    bounds = kpm_params.get('bounds', None)
    if eps <= 0:
        raise ValueError("'eps' must be positive")
    # Hamiltonian rescaled as in Eq. (24)
    _, (_a, _b) = kwant.kpm._rescale(H0[one], eps=eps, bounds=bounds, v0=None)
    # rescale as sparse matrix
    H0[one] = (H0[one] - _b * scipy.sparse.identity(H0[one].shape[0], dtype='complex', format='csr')) / _a
    H1 /= _a
    # extract the number of moments or set default to 100
    energy_resolution = kpm_params.get('energy_resolution')
    if energy_resolution is not None:
        num_moments = int(np.ceil((1.6 * _a) / energy_resolution))
        if kpm_params.get('num_moments'):
            raise TypeError("Only one of 'num_moments' or 'energy_resolution' can be provided.")
    elif kpm_params.get('num_moments') is None:
        num_moments = 100
    else:
        num_moments = kpm_params.get('num_moments')

    ham = H0 + H1
    N = ham.shape[0]

    num_vectors = kpm_params.get('num_vectors', 10)
    vectors = kpm_params.get('vector_factory')
    if vectors is None:
        # Make random vectors
        rng = ensure_rng(kpm_params.get('rng'))
        vectors = np.exp(2j * np.pi * rng.random_sample((num_vectors, N)))
    if not isinstance(vectors, PerturbativeModel):
        vectors = PerturbativeModel({1: np.atleast_2d(vectors)})

    _kernel = kpm_params.get('kernel', kwant.kpm.jackson_kernel)
    operator = kpm_params.get('operator')

    # Calculate all the moments, this is where most of the work is done.
    moments = []
    for kpm_vec in _perturbative_kpm_vector_generator(ham, vectors, num_moments):
        next_moment = vectors.conj() * (operator * kpm_vec.T() if operator else kpm_vec.T())
        if trace:
            next_moment = next_moment.trace() / num_vectors
        moments.append(next_moment)

    def expansion(f):
        """
        Perturbative expansion of `Tr(f(H) * A)` using stochastic trace.
        The moments are precalculated, so evaluation should be fast.
        """
        coef = np.polynomial.chebyshev.chebinterpolate(lambda x: f(x * _a + _b), num_moments)
        coef = _kernel(coef)
        return sum(c * moment for c, moment in zip(coef, moments))

    return expansion


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
        alpha = PerturbativeModel({1: np.atleast_2d(vectors)})
    else:
        alpha = vectors
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
        # Multiplying sparse matrix with number is much slower
        alpha_next = ham * (2 * alpha) - alpha_prev
        alpha_prev = alpha
        alpha = alpha_next
        yield alpha.T()
        n += 1
