from functools import reduce
from operator import mul
from itertools import product
from numbers import Number
import scipy.sparse
from math import factorial
import numpy as np
import sympy

from .perturbationS import Y_i
from .kpm_funcs import build_greens_function
from .perturbative_model import PerturbativeModel, allclose

one = sympy.sympify(1)


### TODO clean this up, we are not using additional_keys
def get_maximum_powers(basic_keys, max_order=2, additional_keys=None):
    """ Generating list of interesting keys.

    It helps minimize total time of calculations.

    Input
    -----
    basic_keys: array of keys for calculating maximum powers
    max_order: int (maximum momentum order)
    additional_keys: array of tuples (symbol, max_power)

    Note:
    additional key will only appeard in maximal power specified for it.
    """

    # First getting momenta. Total power <= max_order
    sets = [range(max_order+1) for i in range(len(basic_keys))]
    momenta = []
    ### TODO there should be a more efficient way to do the combinatorics here
    for power in product(*sets):
        if sum(power) <= max_order:
            momentum = reduce(mul, [k**n for (k,n) in zip(basic_keys, power)])
            momenta.append(momentum)

    if not additional_keys:
        return momenta

    # Getting additional keys. Power of every key <= max_order
    addition = [[k**n for n in range(m+1)] for (k,m) in additional_keys]
    addition = [reduce(mul, key) for key in product(*addition)]

    output = []
    for (a,b) in product(momenta, addition):
        output.append(a*b)

    return output


def _divide_by_energies(Y_AB, energies_A, vectors_A,
                        energies_B, vectors_B, H_0, kpm_params, precalculate_moments):
    """Apply energy denominators using hybrid KPM Green's function"""
    S_AB = Y_AB.copy()
    # Apply Green's function from the right to Y_AB row by row
    for key, val in S_AB.items():
        # if all of the B subspace is explicitely given, skip KPM
        if len(energies_A) + len(energies_B) < vectors_A.shape[0]:
            # Project out A subspace and the explicit part of B subspace
            val_KPM = (val - val.dot(vectors_A).dot(vectors_A.T.conj())
                       - val.dot(vectors_B).dot(vectors_B.T.conj()))
            # This way we do all vectors for all energies, uses a bit more RAM than
            # absolutely necessary, but makes the code simpler.
            vec_G_Y = build_greens_function(H_0,
                                            params=None,
                                            vectors=val_KPM.conj(),
                                            kpm_params=kpm_params,
                                            precalculate_moments=precalculate_moments)(energies_A)
            res = np.vstack([vec_G_Y[m, m, :].conj() for m in range(len(energies_A))])
        else:
            res = np.zeros(val.shape, dtype=complex)
        # Add back the explicit part
        val_ml = val.dot(vectors_B)
        G_ml = 1/(np.array([energies_A]).T - energies_B)
        res += (val_ml * G_ml).dot(vectors_B.T.conj())
        # Apply projector from right, not really necessary, but safeguards
        # against numerical errors creeping the result into `A` subspace.
        S_AB[key] = res - res.dot(vectors_A).dot(vectors_A.T.conj())
    return S_AB


def _block_commute_diag(H, S):
    # Commutator `[H, S]` written out in block form
    # for block-diagonal `H = ((H_AA, 0), (0, H_BB))`
    # and off-diagonal `S = ((0, S_AB), (S_BA, 0))`.
    ((H_AA, _), (_, H_BB)) = H
    ((_, S_AB), (S_BA, _)) = S
    res_AB = H_AA * S_AB - S_AB * H_BB
    res_BA = H_BB * S_BA - S_BA * H_AA
    return ((0, res_AB), (res_BA, 0))


def _block_commute_AA(H, S):
    # Commutator `[H, S]` written out in block form
    # for off-diagonal `H = ((0, H_AB), (H_BA, 0))`
    # and off-diagonal `S = ((0, S_AB), (S_BA, 0))`.
    # Only the AA block is kept
    ((_, H_AB), (H_BA, _)) = H
    ((_, S_AB), (S_BA, _)) = S
    res_AA = H_AB * S_BA - S_AB * H_BA
    return res_AA


def _block_commute_2(H, S):
    # Commutator `[[H, S], S]` written out in block form
    # for off-diagonal `H = ((0, H_AB), (H_BA, 0))`
    # and off-diagonal `S = ((0, S_AB), (S_BA, 0))`.
    ((_, H_AB), (H_BA, _)) = H
    ((_, S_AB), (S_BA, _)) = S
    res_AA = H_AB * S_BA - S_AB * H_BA
    # Ordering that avoids calculating BB blocks
    res_AB = res_AA * S_AB - (S_AB * H_BA) * S_AB + (S_AB * S_BA) * H_AB
    res_BA = H_BA * (S_AB * S_BA) - S_BA * (H_AB * S_BA) - S_BA * res_AA
    return ((0, res_AB), (res_BA, 0))

def get_effective_model(H0, H1, evec_A, evec_B=None, order=2, interesting_keys=None,
                        kpm_params=None, _precalculate_moments=False):
    """Return effective model for given perturbation.

    Implementation of quasi-degenerated perturbation theory.
    Inspired by appendix in R. Winkler book.

    Parameters
    ----------
    H0 : array or PerturbativeModel
        Unperturbed hamiltonian, dense or sparse matrix. If
        provided as PerturbativeModel, it must contain a single
        entry {sympy.sympify(1): array}
    H1 : dict of {sympy.Symbol : array} or PerturbativeModel
        Perturbation to the Hamiltonian
    evec_A : array
        Basis of the interesting `A` subspace of `H0` given
        as a set of orthonormal column vectors
    evec_B : array or None (default)
        Basis of a subspace of the `B` subspace of `H0` given
        as a set of orthonormal column vectors, which will be
        taken into account exactly in hybrid-KPM approach.
        If `evec_A` and `evec_B` contain all eigenvectors of
        H0, everything is treated exactly and KPM is not used.
        If the Hamiltonian is 2x2, must provide `evec_B`.
    order : int (default 2)
        Order of the perturbation calculation.
    interesting_keys : iterable of sympy expressions or None (default)
        By default up to `order` order polynomials of every key in `H1`
        is kept. If not all of these are interesting, tha calculation
        can be sped up by providing a subset of these keys to keep.
        Should be a subset of the keys up to `order` and should contain
        all subexpressions of desired keys, as terms not listed in
        `interesting_keys` are discarded at every step of the calculation.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
        By default num_moments=100 and Jackson kernel is used.
    _precalculate_moments : bool, default False
        Whether to precalculate and store all the KPM moments.
        Typically the default is the best choice for effective model
        calculation.

    Returns
    -------
    Hd : PerturbativeModel
        Effective Hamiltonian in the `A` subspace.
    """

    ### TODO make sure interesting_keys is a subset of the default list,
    # as higher powers might appear but will not contain all contributions.
    if interesting_keys is None:
        interesting_keys = get_maximum_powers(H1.keys(), order)

    if order > len(Y_i) + 1:
        raise ValueError('Terms for {}\'th order perturbation theory not available. '
                         'If you want to calculate {}\'th order perturbations, run '
                         'generating_s_terms.ipynb with wanted_order = {}. '
                         'This may take very long.'.format(order, order, order-1))

    # Convert to appropriate format
    if not isinstance(H0, PerturbativeModel):
        H0 = PerturbativeModel({one: H0}, interesting_keys=interesting_keys)
    elif not (len(H0) == 1 and list(H0.keys()).pop() == one):
        raise ValueError('H0 must contain a single entry {sympy.sympify(1): array}.')

    if evec_A.shape[1] < H0.shape[0] <= 2 and evec_B is None:
        raise ValueError('If the Hamiltonian is 2x2, must provide `evec_B`.')

    H0 = H0.tosparse()
    H1 = PerturbativeModel(H1, interesting_keys=interesting_keys)
    H1 = H1.tosparse()
    if isinstance(evec_A, scipy.sparse.spmatrix):
        evec_A = evec_A.A

    if evec_B is None:
        evec_B = np.empty((evec_A.shape[0], 0))
        ev_B = []
    else:
        if isinstance(evec_B, scipy.sparse.spmatrix):
            evec_B = evec_B.A
        H0_BB = evec_B.T.conj() * H0 * evec_B
        ev_B = np.diag(H0_BB[one])
        if not (allclose(np.diag(ev_B), H0_BB[one]) and
                allclose(evec_B.T.conj() @ evec_B, np.eye(evec_B.shape[1]))):
            raise ValueError('evec_B must be orthonormal eigenvectors of H0')
        if not allclose(evec_B.T.conj() @ evec_A, 0):
            raise ValueError('Vectors in evec_B must be orthogonal to all vectors in evec_A.')

    # Generate projected terms
    H0_AA = evec_A.T.conj() * H0 * evec_A
    ev_A = np.diag(H0_AA[one])
    if not (allclose(np.diag(ev_A), H0_AA[one]) and
            allclose(evec_A.T.conj() @ evec_A, np.eye(evec_A.shape[1]))):
        raise ValueError('evec_A must be orthonormal eigenvectors of H0')
    H1_AA = evec_A.T.conj() * H1 * evec_A
    assert H1_AA == H1_AA.T().conj()
    H2_AB = evec_A.T.conj() * H1 - H1_AA * evec_A.T.conj()
    H2_BA = H1 * evec_A - evec_A * H1_AA
    assert H2_AB == H2_BA.T().conj()
    assert not any((H0_AA.issparse(), H1_AA.issparse(), H2_AB.issparse(), H2_BA.issparse()))

    # Generate `S` to `order-1` order
    S_AB = []
    S_BA = []
    for i in range(1, order):
        # `Y_i` is the right hand side of the equation `[H0, S_i] = Y_i`.
        # We take the expressions for `Y_i` from an external file.
        Y = Y_i[i - 1]
        Y_AB = Y(H0_AA, H0, H1_AA, H1, H2_AB, H2_BA, S_AB, S_BA)
        # Solve for `S_i` by applying Green's function
        S_AB_i = _divide_by_energies(Y_AB, ev_A, evec_A, ev_B, evec_B,
                                     H0[one], kpm_params=kpm_params,
                                     precalculate_moments=_precalculate_moments)
        S_BA_i = -S_AB_i.T().conj()
        assert not any((Y_AB.issparse(), S_AB_i.issparse(), S_BA_i.issparse()))
        S_AB.append(S_AB_i)
        S_BA.append(S_BA_i)
    S_AB = sum(S_AB)
    S_BA = -S_AB.T().conj()
    S = ((0, S_AB), (S_BA, 0))

    # Generate effective Hamiltonian `Hd` to `order` order using `S`.
    # 0th commutators of H
    comm_diag = ((H0_AA + H1_AA, 0), (0, H0 + H1))
    comm_offdiag = ((0, H2_AB), (H2_BA, 0))
    # Add 0th commutator of diagonal
    Hd = H0_AA + H1_AA
    # Add 1st commutator of off-diagonal
    Hd += _block_commute_AA(comm_offdiag, S)
    # Make 1st commutator of diagonal
    comm_diag = _block_commute_diag(comm_diag, S)
    # Add 2nd commutator of diagonal
    Hd += _block_commute_AA(comm_diag, S) * (1 / factorial(2))
    assert Hd == Hd.T().conj(), Hd.todense()

    for j in range(2, order//2 + 1):
        # Make (2j-2)'th commutator of off-diagonal
        comm_offdiag = _block_commute_2(comm_offdiag, S)
        # Add (2j-1)'th commutator of off-diagonal
        Hd += _block_commute_AA(comm_offdiag, S) * (1 / factorial(2*j-1))
        # Make (2j-1)'th commutator of diagonal
        comm_diag = _block_commute_2(comm_diag, S)
        # Add 2j'th commutator of diagonal
        Hd += _block_commute_AA(comm_diag, S) * (1 / factorial(2*j))
        assert not Hd.issparse()
        assert Hd == Hd.T().conj(), Hd.todense()

    return Hd
