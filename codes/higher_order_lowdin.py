from .perturbationS import Y_i

import collections

from functools import reduce
from operator import mul
from itertools import product
import scipy.sparse
from math import factorial
import sympy
from sympy.core.basic import Basic
import numpy as np
from .kpm_funcs import build_greens_function
from numbers import Number
from .qsymm.model import Model, allclose, _find_shape, _find_momenta

one = sympy.sympify(1)


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
                        energies_B, vectors_B, H_0, kpm_params):
    """Apply energy denominators using hybrid KPM Green's function"""
    S_AB = Y_AB.copy()
    # Apply Green's function from the right to Y_AB row by row
    for key, val in S_AB.items():
        # if all of the B subspace is explicitely given, skip KPM
        if len(energies_A) + len(energies_B) < vectors_A.shape[0]:
            # Project out A subspace and the explicit part of B subspace
            val_KPM = (val - val.dot(vectors_A).dot(vectors_A.T.conj())
                       - val.dot(vectors_B).dot(vectors_B.T.conj()))
            # This way we do it for all rows at once, bit faster but uses more RAM
            vec_G_Y = build_greens_function(H_0,
                                            params=None,
                                            vectors=val_KPM.conj(),
                                            kpm_params=kpm_params)
            res = np.vstack([vec_G_Y(E_m)[m].conj() for m, E_m in enumerate(energies_A)])
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

def get_effective_model(H0, H1, evec_A, evec_B=None, interesting_keys=None, order=2, kpm_params=None):
    """Return effective model for given perturbation.

    Implementation of quasi-degenerated perturbation theory.
    Inspired by appendix in R. Winkler book.

    Parameters
    ----------
    H0 : array
        Unperturbed hamiltonian, dense or sparse matrix
    H1 : dict of {sympy.Symbol : array}
        Perturbation to the Hamiltonian
    evec_A : array
        Basis of the interesting `A` subspace of H0 given
        as a set of orthonormal column vectors
    evec_B : array
        Basis of a subspace of the `B` subspace of H0 given
        as a set of orthonormal column vectors, which will be
        taken into account exactly in hybrid-KPM approach.
    interesting_keys : list of sympy.Symbol
        List of interesting keys to keep in the calculation.
        Should contain all subexpressions of desired keys, as
        terms not listed in `interesting_keys` are discarded
        at every step of the calculation. By default up to
        `order` power of every key in H1 is kept.
    order : int
        Order of the perturbation calculation
    kpm_params : dict
        Parameters to pass on to KPM solver. By default num_moments=100.

    Returns
    -------
    Hd : PerturbativeModel
        Effective Hamiltonian in the `A` subspace.
    """

    if interesting_keys is None:
        interesting_keys = get_maximum_powers(H1.keys(), order)

    if order > len(Y_i) + 1:
        raise ValueError('Terms for {}\'th order perturbation theory not available. '
                         'If you want to calculate {}\'th order perturbations, run '
                         'generating_s_terms.ipynb with wanted_order = {}. '
                         'This may take very long.'.format(order, order, order-1))

    # Convert to appropriate format
    H0 = PerturbativeModel({one: H0}, interesting_keys=interesting_keys)
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
                                     H0[one], kpm_params=kpm_params)
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


# *********************** POLYNOMIAL CLASS ************************************

# Functions to handle different types of arrays
# If either of them are dense, the result is dense.
def _smart_dot(a, b):
    if isinstance(a, scipy.sparse.spmatrix) or isinstance(b, scipy.sparse.spmatrix):
        return scipy.sparse.csr_matrix.dot(a, b)
    else:
        return np.dot(a, b)

def _smart_add(a, b):
    if isinstance(a, scipy.sparse.spmatrix) and isinstance(b, scipy.sparse.spmatrix):
        return a + b
    elif isinstance(a, scipy.sparse.spmatrix):
        return (a + b).A
    elif isinstance(b, scipy.sparse.spmatrix):
        return (b + a).A
    else:
        return a + b


class PerturbativeModel(Model):

    def __init__(self, hamiltonian={}, locals=None, interesting_keys=None, momenta=[]):
        """General class to store Hamiltonian families.
        Can be used to efficiently store any matrix valued function.
        Implements many sympy and numpy methods. Arithmetic operators are overloaded,
        such that `*` corresponds to matrix multiplication.
        Enhances the functionality of Model by allowing interesting_keys to be
        specified, terms with different coefficients are discarded.
        Allows the use of scipy sparse matrices besides numpy arrays.

        Parameters
        ----------
        hamiltonian : str, SymPy expression or dict
            Symbolic representation of a Hamiltonian.  It is
            converted to a SymPy expression using `kwant_continuum.sympify`.
            If a dict is provided, it should have the form
            {sympy expression: array} with all arrays either dense or sparse, with
            the same size and sympy expressions consisting purely of symbolic
            coefficients, no constant factors.
        locals : dict or ``None`` (default)
            Additional namespace entries for `~kwant_continuum.sympify`.  May be
            used to simplify input of matrices or modify input before proceeding
            further. For example:
            ``locals={'k': 'k_x + I * k_y'}`` or
            ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
        interesting_keys : iterable of sympy expressions
            Set of symbolic coefficients that are kept, anything that does not
            appear here is discarded. Useful for perturbative calculations where
            only terms to a given order are needed.
        momenta : list of int or list of Sympy objects
            Indices of momenta the monomials depend on from 'k_x', 'k_y' and 'k_z'
            or a list of names for the momentum variables.
        """
        # Usual case is initializing with a dict
        if isinstance(hamiltonian, dict):
            collections.UserDict.__init__(self, hamiltonian)
            self.shape = _find_shape(hamiltonian)
            self.momenta = _find_momenta(momenta)
        # Otherwise try to parse the input with Model's machinery.
        # This will always result in a dense PerturbativeModel.
        else:
            super().__init__(hamiltonian, locals, momenta)

        if interesting_keys is not None:
            self.interesting_keys = set(interesting_keys)
        else:
            self.interesting_keys = set()
        # Removing all key that are not interesting.
        if self.interesting_keys:
            for key in self.keys():
                if key not in self.interesting_keys:
                    del self[key]

    def tosympy(self, digits=12):
        """Convert into sympy matrix."""
        result = self.todense()
        result = [(key * np.round(val, digits)) for key, val in result.items()]
        result = sympy.Matrix(sum(result))
        result.simplify()
        return result

    def tosparse(self):
        output = self.copy()
        for key, val in output.items():
            output[key] = scipy.sparse.csr_matrix(val)
        return output

    def issparse(self):
        for key, val in self.items():
            if isinstance(val, scipy.sparse.spmatrix):
                return True
        return False

    def todense(self):
        output = self.copy()
        for key, val in output.items():
            if isinstance(val, scipy.sparse.spmatrix):
                output[key] = val.A
        return output

    def lambdify(self, *gens):
        """Lambdify using gens as variables."""
        result = self.tosympy()
        result = sympy.lambdify(gens, result, 'numpy')
        return result

    def evalf(self, subs=None):
        result = []
        for key, val in self.items():
            key = float(key.evalf(subs=subs))
            result.append(key * val)
        return sum(result)

    def __eq__(self, other):
        a = self.todense()
        b = other.todense()
        for key in a.keys() | b.keys():
            if not allclose(a[key], b[key]):
                return False
        return True

    def __add__(self, other):
        # Addition of Models. It is assumed that both Models are
        # structured correctly, every key is in standard form.
        # Define addition of 0 and {}
        if not other:
            result = self.copy()
        # If self is empty return other
        elif not self and isinstance(other, type(self)):
            result = other.copy()
        elif isinstance(other, type(self)):
            result = self.copy()
            for key, val in other.items():
                result[key] = _smart_add(result[key], val)
        else:
            raise NotImplementedError('Addition of {} with {} not supported'.format(type(self), type(other)))
        return result

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        if isinstance(other, Number):
            result = self.copy()
            for key, val in result.items():
                result[key] *= other
        elif isinstance(other, Basic):
            result = PerturbativeModel({key * other: val for key, val in self.items()},
                                        interesting_keys=interesting_keys)
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(val, other)
            result.shape = (self.shape[0], other.shape[1])
        elif isinstance(other, PerturbativeModel):
            interesting_keys = self.interesting_keys | other.interesting_keys
            result = sum([PerturbativeModel({k1 * k2: _smart_dot(v1, v2)}, interesting_keys=interesting_keys)
                      for (k1, v1), (k2, v2) in product(self.items(), other.items())
                      if (k1 * k2 in interesting_keys or not interesting_keys)])
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __truediv__(self, B):
        result = self.copy()

        if isinstance(B, PerturbativeModel):
            raise TypeError(
                "unsupported operand type(s) for /: 'PerturbativeModel' and "
                "'PerturbativeModel'"
            )
        else:
            for key in result:
                result[key] /= B

        return result

    def __rmul__(self, other):
        # Left multiplication by numbers, sympy symbols and arrays
        if isinstance(other, Number):
            result = self.__mul__(other)
        elif isinstance(other, Basic):
            result = PerturbativeModel({other * key: val for key, val in self.items()},
                                        interesting_keys=interesting_keys)
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(other, val)
            result.shape = (other.shape[0], self.shape[1])
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def around(self, decimals=3):
        raise NotImplementedError()
