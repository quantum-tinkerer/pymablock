import numpy as np
import sympy
import scipy.sparse

import kwant.kpm
from kwant._common import ensure_rng

from .qsymm.model import Model

from .higher_order_lowdin import _interesting_keys
from .kpm_funcs import _kpm_preprocess

one = sympy.sympify(1)


def trace_perturbation(H0, H1, order=2, interesting_keys=None,
                       trace=True, kpm_params=dict()):
    """
    Perturbative expansion of `Tr(f(H) * A)` using stochastic trace.

    Parameters
    ----------

    H0 : array or Model
        Unperturbed hamiltonian, dense or sparse matrix. If
        provided as Model, it must contain a single
        entry {1 : array}.
    H1 : dict of {symbol : array} or Model
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
        operator : 2D array or Model or None (default)
            Operator in the expectation value, default is identity.
        vector_factory : 1D or 2D array or Model or None (default)
            Vector of length `N` or array of vectors with shape `(M, N)`.
            If Model, must be of shape `(M, N)`. The size of
            the last index should be the same as the size of the Hamiltonian.
            By default, random vectors are used.

    Returns
    -------

    expansion : callable
        A function that takes the function of energy `f` and returns the
        expansion of the trace. This evaluation is very fast, as the
        moments of the expansion are precalculated.
    """
    H1 = Model(H1)
    all_keys = _interesting_keys(H1.keys(), order)
    if interesting_keys is None:
        interesting_keys = all_keys
    else:
        interesting_keys = set(interesting_keys)
    H1.keep = interesting_keys
    if not interesting_keys <= all_keys:
        raise ValueError('`interesting_keys` should be a subset of all monomials of `H1.keys()` '
                         'up to total power `order`.')

    # Convert to appropriate format
    if not isinstance(H0, Model):
        H0 = Model({1: H0}, keep=interesting_keys)
    elif not (len(H0) == 1 and list(H0.keys()).pop() == 1):
        raise ValueError('H0 must contain a single entry {sympy.sympify(1): array}.')
    # Find the bounds of the spectrum and rescale `ham`
    H0[one], (_a, _b), num_moments, _kernel = _kpm_preprocess(H0[one], kpm_params)
    H1 /= _a

    ham = H0 + H1
    N = ham.shape[0]

    num_vectors = kpm_params.get('num_vectors', 10)
    vectors = kpm_params.get('vector_factory')
    if vectors is None:
        # Make random vectors
        rng = ensure_rng(kpm_params.get('rng'))
        vectors = np.exp(2j * np.pi * rng.random_sample((num_vectors, N)))
    if not isinstance(vectors, Model):
        vectors = Model({1: np.atleast_2d(vectors)})

    operator = kpm_params.get('operator')

    # Calculate all the moments, this is where most of the work is done.
    moments = []
    # Precalculate operator acting on vectors, assume it is Hermitian
    if operator:
        op_vecs = (operator @ vectors.T()).T()
    else:
        op_vecs = vectors

    for kpm_vec in _perturbative_kpm_vectors(ham, vectors, num_moments):
        next_moment = op_vecs.conj() @ kpm_vec.T()
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


def _perturbative_kpm_vectors(ham, vectors, max_moments):
    """
    Generator object that succesively yields KPM vectors `T_n(ham) |vector>`
    for vectors in `vectors` for n in [0, max_moments].

    Parameters
    ----------
    ham : Model
        Hamiltonian, Model with shape `(N, N)`.
        The `keep` property is used to limit the powers of
        free parameters that we keep track of in the expansion.
    vectors : 1D or 2D array or Model
        Vector of length `N` or array of vectors with shape `(M, N)`.
        If Model, must be of shape `(M, N)`. The size of
        the last index should be the same as the size of `ham` `N`.
    max_moments : int
        Number of moments to stop with iteration

    Notes
    -----
    Returns a sequence of expanded vectors, Models of shape (M, N).
    If the input was a vector, M=1.
    """
    if not isinstance(vectors, Model):
        alpha = Model({1: np.atleast_2d(vectors)})
    else:
        alpha = vectors
    # Internally store as column vectors
    alpha = alpha.T()
    n = 0
    yield alpha.T()
    n += 1
    alpha_prev = alpha
    alpha = ham @ alpha
    yield alpha.T()
    n += 1
    while n < max_moments:
        alpha_next = 2 * ham @ alpha - alpha_prev
        alpha_prev = alpha
        alpha = alpha_next
        yield alpha.T()
        n += 1
