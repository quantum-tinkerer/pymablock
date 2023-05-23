from functools import reduce
from operator import mul

import numpy as np
import sympy

from kwant._common import ensure_rng

from qsymm.model import Model, _symbol_normalizer

from .kpm_funcs import rescale
from kwant.kpm import jackson_kernel

one = sympy.sympify(1)


def _interesting_keys(keys, order=2):
    """
    Generate list of interesting keys as monomials of `keys`
    with maximum total power `order`.
    It helps minimize total time of calculations.

    Parameters
    ----------
    keys: iterable of sympy expressions
        Keys that appear in monomials.
    order: int (default 2)
        Maximum total power of `keys` in `interesting_keys`.

    Returns
    -------
    interesting_keys: set of sympy expressions
    """

    def partition(n, d, depth=0):
        # Partition the number n into d parts sensitive to the order of terms
        if d == depth:
            return [[]]
        return [
            item + [i]
            for i in range(n + 1)
            for item in partition(n - i, d, depth=depth + 1)
        ]

    # Generate partitioning of `order` to `len(keys) + 1` parts, this includes all
    # monomials with lower total power as well
    powers = partition(order, len(keys) + 1)
    interesting_keys = set(
        reduce(mul, [k**n for (k, n) in zip(keys, power)]) for power in powers
    )
    return interesting_keys


def trace_perturbation(
    H0,
    H1,
    order=2,
    interesting_keys=None,
    trace=True,
    num_moments=100,
    operator=None,
    num_vectors=10,
    vectors=None,
    normalized_vectors=False,
    kpm_params=dict(),
):
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
        Order of the perturbation calculation, up to `order` order
        polynomials of every key in `H1` is kept during the expansion.
        If `interesting_keys` is provided, `order` is ignored.
    interesting_keys : iterable of sympy expressions or None (default)
        The calculation can be sped up by providing an explicit set of keys
        to keep. Should contain all subexpressions of desired keys, as terms
        not listed in `interesting_keys` are discarded at every step of the
        calculation.
    trace : bool, default True
        Whether to calculate the trace. If False, all matrix elements
        between pairs of vectors are separately returned.
    num_moments : int, default 100
        Number of moments in the KPM expansion.
    operator : 2D array or Model or None (default)
        Operator in the expectation value, default is identity.
    num_vectors : int, default 10
        Number of random vectors used in KPM. Ignored if `vector_factory`
        is provided.
    vectors : 1D or 2D array or Model or None (default)
        Vector of length `N` or array of vectors with shape `(M, N)`.
        If Model, must be of shape `(M, N)`. The size of
        the last index should be the same as the size of the Hamiltonian.
        By default, random vectors are used.
    normalized_vectors: bool, default False
        Whether the vectors returned by `vector_factory` are normalized.
        By default the result is divided by the number of vectors, as
        random phase vectors are assumed. If True, the result is not
        divided.
    kpm_params : dict, optional
        Dictionary containing additional KPM parameters, see `~kwant.kpm`.

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
        interesting_keys = {_symbol_normalizer(k) for k in interesting_keys}
    H1.keep = interesting_keys

    # Convert to appropriate format
    if not isinstance(H0, Model):
        H0 = Model({1: H0}, keep=interesting_keys)
    elif not (len(H0) == 1 and list(H0.keys()).pop() == 1):
        raise ValueError("H0 must contain a single entry {sympy.sympify(1): array}.")
    # Find the bounds of the spectrum and rescale `ham`
    kpm_params["num_moments"] = num_moments
    H0[one], (_a, _b) = rescale(H0[one], kpm_params.get("eps", 0.05))
    H1 /= _a

    ham = H0 + H1
    N = ham.shape[0]

    if vectors is None:
        # Make random vectors
        rng = ensure_rng(kpm_params.get("rng"))
        vectors = np.exp(2j * np.pi * rng.random_sample((num_vectors, N)))
    if not isinstance(vectors, Model):
        vectors = Model({1: np.atleast_2d(vectors)})

    # Calculate all the moments, this is where most of the work is done.
    moments = []
    # Precalculate operator acting on vectors, assume it is Hermitian
    if operator is not None:
        op_vecs = (operator @ vectors.T()).T()
    else:
        op_vecs = vectors

    for kpm_vec in _perturbative_kpm_vectors(ham, vectors, num_moments):
        next_moment = op_vecs.conj() @ kpm_vec.T()
        if trace:
            next_moment = next_moment.trace()
        if not normalized_vectors:
            next_moment /= num_vectors
        moments.append(next_moment)

    def expansion(f):
        """
        Perturbative expansion of `Tr(f(H) * A)` using stochastic trace.
        The moments are precalculated, so evaluation should be fast.
        """
        coef = np.polynomial.chebyshev.chebinterpolate(
            lambda x: f(x * _a + _b), num_moments
        )
        coef = jackson_kernel(coef)
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
