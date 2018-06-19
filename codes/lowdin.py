from .kpm_funcs import build_perturbation

from .misc import make_commutative, monomials, discretize_with_hoppings
from .qsymm.linalg import simult_diag

import kwant

import sympy
import itertools
import numpy as np
import scipy.linalg as la


# Code responsible for preparation of the initial Hamiltonian

def separate_hamiltonian(ham, gens):
    # First, make sure that generators are commutative symbols
    gens = [kwant.continuum.sympify(g) for g in gens]
    gens = [make_commutative(g, g) for g in gens]

    # Second, make sure that gens in "ham" are commutative
    ham = make_commutative(ham, *gens)

    # Finally, get monomials
    H1 = monomials(ham, gens)
    H0 = H1.pop(1)
    return H0, H1


def prepare_hamiltonian(ham, gens, coords, grid, shape, start, locals=None):
    """Return systems corresponding to H0 and H1 part of full Hamiltonian.

    Parameters
    ----------
    ham : str or SymPy expression
        Symbolic representation of a continuous Hamiltonian.  It is
        converted to a SymPy expression using `kwant.continuum.sympify`.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of the perturbation.
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


def decouple_basis(operators, subspace):
    """Decouple eigenstates in subspace by diagonalizing operators."""
    operators = [inbasis(op, subspace) for op in operators]
    U = np.hstack(simult_diag(operators))

    list_ev = []
    for op in operators:
        ev = np.diag(triproduct(U, op, U)).real
        assert np.allclose(sorted(la.eigvalsh(op)), sorted(ev))
        list_ev.append(ev)

    sort_indices = np.lexsort(list_ev)
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


def sympify_perturbation(M1=None, M2=None, decimals=12):
    terms = []
    terms += [(k, v) for k, v in M1.items() if M1 is not None]
    terms += [(k[0] * k[1], v) for k, v in M2.items() if M2 is not None]

    if len(terms) == 0:
        raise ValueError("At least one of 'M1' or 'M2' should contain "
                         "some items.")
    output = []
    for k, v in terms:
        output.append(k * sympy.Matrix(np.round(v, decimals)))
    return sympy.MatAdd(*output).as_explicit()


# Explicit implementation of perturbation theory

def first_order(perturbation, states):
    """Return first order effective model.

    Parameters
    ----------
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H1.
    states : array(N, M)
        Set of M eigenstates of H0 that make a perturbation basis.
        The numpy convention is used, where "states[:, i]" is i-th
        eigenstate.

    Returns
    -------
    model : dict: symbol -> SymPy expression -> array(M, M)
        First order effective model.
    """
    output = {}
    M = states.shape[1]

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    for symbol in perturbation:
        mat = perturbation[symbol]
        t = np.zeros((M, M), dtype=complex)

        # iterate over states in group A
        for i, j in itertools.product(range(M), range(M)):
            element = triproduct(states[:, i], mat, states[:, j])
            t[i, j] = element

        output[symbol] = t

    return output


def second_order_explicit(perturbation, indices, ev, evec, truncate=True):
    """Return second order effective model.

    Parameters
    ----------
    perturbation : array(N, N) or dict: SymPy expression -> array(N, N)
        Perturbation Hamiltonian H1.
    indices : sequence of integers
        Indices of states from group A
    ev : array(N)
        Energies of H0 for all states in the system.
    evec : array(N, N)
        Eigenstates of H0 for al states in the system.
        The numpy convention is used, where "evec[:, i]" is i-th
        eigenstate.
    truncate : bool
        If "truncate=True" then terms for which total power of symbols > 2
        will not be calculated.
    Returns
    -------
    model : dict: (symbol, symbol) -> SymPy expression -> array(M, M)
        Second order effective model.
    """

    output = {}
    M = len(indices)

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    def calculate_ijm(H1L, H1R, i, j, m):
        """Return 1/2 x H'_{im} H'_{mj} x (1 / (Ei - Em) + 1 / (Ej / Em))."""
        v1 = triproduct(evec[:, i], H1L, evec[:, m])
        v2 = triproduct(evec[:, m], H1R, evec[:, j])
        return 0.5 * v1 * v2 * (1 / (ev[i] - ev[m]) + 1 / (ev[j] - ev[m]))

    def power(expr):
        """Return total power of factors in the expression."""
        powers = [s.as_base_exp()[1] for s in expr.as_ordered_factors()]
        return sum(powers)

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
            for m in range(len(ev)):

                # Make sure we do not count states from group A
                if m in indices:
                    continue

                element += calculate_ijm(H1L, H1R, i, j, m)

            elements.append(element)

        output[SL, SR] = np.array(elements).reshape(M, M)

    return output


# KPM optimisation of second order perturbation theory

def second_order_kpm(hamiltonian, perturbation, energies, subspace,
                     num_moments=1000, truncate=True):
    """Return second order effective model."""

    output = {}
    kpm_params = dict(num_moments=num_moments)

    # Cast perturbation to "dict" if it is "array"
    if not isinstance(perturbation, dict):
        perturbation = {1: perturbation}

    def power(expr):
        """Return total power of factors in the expression."""
        powers = [s.as_base_exp()[1] for s in expr.as_ordered_factors()]
        return sum(powers)

    for SL, SR in itertools.product(perturbation.keys(), repeat=2):

        if truncate and (power(SL) + power(SR) > 2):
            continue

        H1L = perturbation[SL]
        H1R = perturbation[SR]

        element = build_perturbation(energies, subspace.T, hamiltonian,
                                     H1L, H1R, kpm_params=kpm_params)

        output[SL, SR] = element

    return output
