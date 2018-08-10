from .kpm_funcs import build_perturbation

from .misc import make_commutative, monomials, discretize_with_hoppings
from .qsymm.linalg import simult_diag

import kwant

import sympy
import itertools
import collections
import numpy as np
import scipy.linalg as la


# Code responsible for preparation of the initial Hamiltonian

def separate_hamiltonian(ham, gens):
    """Separate "ham" into "H0" and "H1"."""

    # Cast strings to sympy objects using kwant.continuum.sympify rules
    # Elements that are already symbols will not be altered.
    gens = [kwant.continuum.sympify(g) for g in gens]

    # Select commutative generators and make sure that they are commutative in
    # the Hamiltonian
    commutative_gens = [g for g in gens if g.is_commutative]
    ham = make_commutative(ham, *commutative_gens)

    # Now get the monomials and separate H0 from H1
    H1 = monomials(ham, gens)
    try:
        H0 = H1.pop(1)
    except KeyError:
        raise ValueError('Separation of perturbation failed. '
                         'Check if "gens" are chosen correctly.')
    return H0, H1


def prepare_hamiltonian(ham, gens, coords, grid, shape, start, locals=None):
    """Return systems corresponding to H0 and H1 part of full Hamiltonian.

    Parameters
    ----------
    ham : str or SymPy expression
        Symbolic representation of a continuous Hamiltonian.  It is
        converted to a SymPy expression using `kwant.continuum.sympify`.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of the perturbation. If this is a sequence of strings then
        corresponding symbols will be generated using `kwant.continuum.sympify`
        rules, especially regarding the commutative properties. If this is
        already a sequence of SymPy symbols then their commutative properties
        will be respected, i.e if symbol is defined as commutative in "gens" it
        will be casted to the commutative symbol in "ham". Commutative symbols
        will not however be casted to noncommutative.
    coords : sequence of strings, or ``None`` (default)
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.  If not provided, `coords` will be obtained from the input
        Hamiltonian by reading the present coordinates and momentum operators.
    grid : int or float, default: 1
        Spacing of the (quadratic or cubic) discretization grid.
    shape : callable
        A boolean function of site returning whether the site should be
        included in the system or not. The shape must be compatible
        with the system's symmetry.
    start : `Site` instance or iterable thereof or iterable of numbers
        The site(s) at which the the flood-fill starts.  If start is an
        iterable of numbers, the starting site will be
        ``template.closest(start)``.
    locals : dict or ``None`` (default)
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.

    Returns
    -------
    H0: finalized "kwant.system"
    H1: dict: SymPy symbol -> finalized "kwant.system"

    "kwant" systems can be used to built corresponding Hamiltonian matrices
    """

    def _discretize_and_fill(operator, coords, grid, shape, start):
        """Discretize given operator and fill appropriate system.

        Use modified version of "kwant.continuum.discretize" to workaround
        flood-fill algorithm when discretizing operators.
        """
        tb = discretize_with_hoppings(
            operator, coords, grid_spacing=grid
        )
        syst = kwant.Builder()
        syst.fill(tb, shape, start);
        return syst.finalized()

    ham = kwant.continuum.sympify(ham, locals=locals)
    H0, H1 = separate_hamiltonian(ham, gens)

    H0 = _discretize_and_fill(H0, coords, grid, shape, start)
    H1 = {k: _discretize_and_fill(v, coords, grid, shape, start)
          for k, v in H1.items()}

    return H0, H1


# Various helper functions

def triproduct(left, matrix, right):
    """Calculate "vector^dagger @ matrix @ vector" product."""
    return left.T.conjugate() @ matrix @ right


def inbasis(operator, subspace):
    """Return operator in basis of a given subspace."""
    return subspace.T.conjugate() @ operator @ subspace


def power(expr):
    """Return total power of factors in the expression."""
    powers = [s.as_base_exp()[1] for s in expr.as_ordered_factors()]
    return sum(powers)


def decouple_basis(operators, subspace, sorting_decimals=6):
    """Decouple eigenstates in subspace by diagonalizing operators."""
    operators = [inbasis(op, subspace) for op in operators]
    U = np.hstack(simult_diag(operators))

    list_ev = []
    for op in operators:
        ev = np.diag(triproduct(U, op, U)).real
        assert np.allclose(sorted(la.eigvalsh(op)), sorted(ev))
        list_ev.append(ev)

    sort_indices = np.lexsort(np.round(list_ev, sorting_decimals))
    return U[:, sort_indices], [ev[sort_indices] for ev in list_ev]


def apply_smart_gauge(evec):
    """Apply "smart" gauge choice to prettify final output.

    I have no idea how this works, but it works, and is wonderful.
    It is a pure magic!

    This metod modifies "evec" in-place.
    """
    for i, v in enumerate(evec.T):
        phase = np.angle(v @ v)
        evec[:, i] = v * np.exp(-1j*phase/2)


def sympify_perturbation(energies=None, components=None, decimals=12):
    terms = []

    if energies is not None:
        terms += [(1, np.diag(energies))]

    if components is not None:
        for M in components:
            terms += [(k, v) for k, v in M.items()]

    if len(terms) == 0:
        raise ValueError("Provide at least one of 'energies' or 'components'.")

    output = []
    for k, v in terms:
        output.append(k * sympy.Matrix(np.round(v, decimals)))
    return sympy.MatAdd(*output).as_explicit()


# Explicit implementation of perturbation theory

def first_order(perturbation, subspace):
    """Return first order effective model calculated explicitly.

    Parameters
    ----------
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H1.
    subspace : array(N, M)
        Eigenstates of H0 for which the effective model will be calculated.
        The numpy convention is used, where "subspace[:, i]" is i-th
        eigenstate.

    Returns
    -------
    model : dict: symbol -> SymPy expression -> array(M, M)
        First order contribution to the effective model.
    """
    output = {}
    M = subspace.shape[1]

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    for symbol in perturbation:
        mat = perturbation[symbol]
        t = np.zeros((M, M), dtype=complex)

        # iterate over states in group A
        for i, j in itertools.product(range(M), range(M)):
            element = triproduct(subspace[:, i], mat, subspace[:, j])
            t[i, j] = element

        output[symbol] = t

    return output


def second_order(hamiltonian, perturbation, evecA, evecB=None, moments=0,
                 truncate=True):
    """Return second order effective model.

    This calculates contribution to second order effective models
    coming from "evecB" states explicitly. If "moments" is not zero then
    contribution to 2nd order effective model coming from states not included
    in evecA and evecB will be calculated through KPM.

    Parameters
    ----------
    hamiltonian : array(N, N)
        Unperturbated Hamiltonian H0.
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H'.
    evecA : array(N, m)
        Eigenstates of H0 for which we calculate the effective model.
        The numpy convention is used, where "subspace[:, i]" is i-th
        eigenstate.
    evecB : array(N, b)
        Eigenstates of H0 which contribution to the effective model we
        include explicitly.
    moments : int
        Number of kpm moments. If not zero kpm contribution to effective
        model will be included.
    truncate : bool
        If "truncate=True" then terms for which total power of expansion
        coefficients > 2 will not be calculated.

    Returns
    -------
    model : dict: SymPy expression -> array(M, M)
        Second order contribution to the effective model.
    """
    if evecB is not None:
        evec = np.column_stack([evecA, evecB])
        ev = (evec.T.conj() @ hamiltonian @ evec).diagonal().real

        indices = list(range(evecA.shape[1]))
        exp = second_order_explicit(perturbation, ev, evec, indices,
                                    truncate=truncate)
    else:
        evec = evecA
        ev = (evec.T.conj() @ hamiltonian @ evec).diagonal().real

        indices = None
        exp = None

    if moments:
        kpm = second_order_kpm(hamiltonian, perturbation, ev, evec,
                               indices=indices, num_moments=moments,
                               truncate=truncate)
    else:
        kpm = None

    return exp, kpm


def second_order_explicit(perturbation, energies, subspace, indices,
                          truncate=True):
    """Return second order effective model calculated explicitly.

    This calculates contribution to second order effective models
    coming from these states in the "subspace" that are not specified
    by "indices".

    Parameters
    ----------
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H'.
    energies : array(k)
        Energies of H0 for all known states in the system.
    subspace : array(N, k)
        Eigenstates of H0 for all known states in the system.
        The numpy convention is used, where "subspace[:, i]" is i-th
        eigenstate. In special case when "k=N" this is function
        returns the exact result.
    indices : sequence of M integers
        Indices of states for which we calculate the effective model.
    truncate : bool
        If "truncate=True" then terms for which total power of expansion
        coefficients > 2 will not be calculated.

    Returns
    -------
    model : dict: SymPy expression -> array(M, M)
        Second order contribution to the effective model.
    """
    M = len(indices)
    output = collections.defaultdict(lambda: np.zeros((M, M), dtype=complex))

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    def calculate_ijm(H1L, H1R, i, j, m):
        """Return 1/2 x H'_{im} H'_{mj} x (1 / (Ei - Em) + 1 / (Ej / Em))."""
        v1 = triproduct(subspace[:, i], H1L, subspace[:, m])
        v2 = triproduct(subspace[:, m], H1R, subspace[:, j])
        c1 = 1 / (energies[i] - energies[m])
        c2 = 1 / (energies[j] - energies[m])
        return 0.5 * v1 * v2 * (c1 + c2)

    for SL, SR in itertools.product(perturbation.keys(), repeat=2):

        if truncate and (power(SL) + power(SR) > 2):
            continue

        H1L = perturbation[SL]
        H1R = perturbation[SR]

        elements = []
        # iterate over states in group A
        for i, j in itertools.product(indices, indices):

            element = 0
            # iterate over states in group B
            for m in range(len(energies)):

                # Make sure we do not count states from group A
                if m in indices:
                    continue

                element += calculate_ijm(H1L, H1R, i, j, m)

            elements.append(element)

        output[SL * SR] += np.array(elements).reshape(M, M)

    return dict(output)


# KPM optimisation of second order perturbation theory

def second_order_kpm(hamiltonian, perturbation, energies, subspace,
                     indices=None, num_moments=1000, truncate=True):
    """Return second order effective model calculated through kpm.

    This calculates contribution to second order effective models
    coming from all states of "hamiltonian" that are not included
    in the "subspace".

    Parameters
    ----------
    hamiltonian : array(N, N)
        Unperturbated Hamiltonian H0.
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H1.
    energies : array(k)
        Energies of "hamiltonian" for all known states.
    subspace : array(N, k)
        Eigenstates of "hamiltonian" for all states known exactly.
        The numpy convention is used, where "subspace[:, i]" is i-th
        eigenstate.
    indices : sequence of M integers
        Indices of states for which we calculate the effective model.
        If unset (None) then all states in "subspace" will be considered.
    truncate : bool
        If "truncate=True" then terms for which total power of symbols > 2
        will not be calculated.
    Returns
    -------
    model : dict: SymPy expression -> array(M, M)
        Second order effective model.
    """

    if indices is None:
        indices = range(len(energies))

    M = len(indices)
    output = collections.defaultdict(lambda: np.zeros((M, M), dtype=complex))
    kpm_params = dict(num_moments=num_moments)

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    for SL, SR in itertools.product(perturbation.keys(), repeat=2):

        if truncate and (power(SL) + power(SR) > 2):
            continue

        H1L = perturbation[SL]
        H1R = perturbation[SR]

        element = build_perturbation(energies, subspace, hamiltonian,
                                     H1L, H1R, indices=indices,
                                     kpm_params=kpm_params)

        output[SL * SR] += element

    return dict(output)
