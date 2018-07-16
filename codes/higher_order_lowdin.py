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
        # Project out A subspace and the explicit part of B subspace
        val_KPM = (val - val.dot(vectors_A).dot(vectors_A.T.conj())
                   - val.dot(vectors_B).dot(vectors_B.T.conj()))
        # This way we do it for all rows at once, bit faster but uses more RAM
        vec_G_Y = build_greens_function(H_0,
                                        params=None,
                                        vectors=val_KPM.conj(),
                                        kpm_params=kpm_params)
        res = np.vstack([vec_G_Y(E_m)[m].conj() for m, E_m in enumerate(energies_A)])
        # Add back the explicit part
        # for E_l, vec_l in zip(energies_B, vectors_B.T):
        #     res += np.vstack([val[m].dot(vec_l.T) * vec_l.conj()/(E_m - E_l)
        #                       for m, E_m in enumerate(energies_A)])
        val_ml = val.dot(vectors_B)
        G_ml = 1/(np.array([energies_A]).T - energies_B)
        res += (val_ml * G_ml).dot(vectors_B.T.conj())
        # Apply projector from right, not really necessary, but safeguards
        # against numerical errors creeping the result into `A` subspace.
        S_AB[key] = res - res.dot(vectors_A).dot(vectors_A.T.conj())
    return S_AB


def _block_commute(H, S_AB, S_BA):
    # Commutator `[H, S]` written out in block form
    # for general `H = ((H_AA, H_AB), (H_BA, H_BB))`
    # and off-diagonal `S = ((0, S_AB), (S_BA, 0))`.
    ((H_AA, H_AB), (H_BA, H_BB)) = H
    res_AB = H_AA * S_AB - S_AB * H_BB
    res_BA = H_BB * S_BA - S_BA * H_AA
    res_AA = H_AB * S_BA - S_AB * H_BA
    res_BB = H_BA * S_AB - S_BA * H_AB
    return ((res_AA, res_AB), (res_BA, res_BB))


def get_effective_model(H0, H1, evec_A, evec_B=None, interesting_keys=None, order=2, kpm_params=None):
    """Return effective model for given perturbation.

    Implementation of quasi-degenerated perturbation theory.
    Inspired by appendix in R. Winkler book.

    Input
    -----
    H0 : array
        Unperturbed hamiltonian, dense or sparse matrix
    H1 : dict of {sympy.Symbol : array}
        Perturbation to the Hamiltonian
    evec_A : array
        Basis of the interesting `A` subspace of H0 given
        as a set of column vectors
    evec_B : array
        Basis of a subspace of the `B` subspace of H0 given
        as a set of column vectors, which will be taken
        into account exactly in hybrid-KPM approach.
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

    Returns:
    --------
    Hd : MatCoeffPolynomial
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
    H0 = MatCoeffPolynomial({1: H0}, interesting_keys=interesting_keys)
    H0 = H0.tosparse()
    H1 = MatCoeffPolynomial(H1, interesting_keys=interesting_keys)
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
        ev_B = np.diag(H0_BB[1])
        assert np.allclose(np.diag(ev_B), H0_BB[1]), 'evec_B should be eigenvectors of H0'

    # Generate projected terms
    H0_AA = evec_A.T.conj() * H0 * evec_A
    ev_A = np.diag(H0_AA[1])
    assert np.allclose(np.diag(ev_A), H0_AA[1]), 'evec_A should be eigenvectors of H0'
    H1_AA = evec_A.T.conj() * H1 * evec_A
    assert H1_AA == H1_AA.H()
    H2_AB = evec_A.T.conj() * H1 - H1_AA * evec_A.T.conj()
    H2_BA = H1 * evec_A - evec_A * H1_AA
    assert H2_AB == H2_BA.H()
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
                                     H0[1], kpm_params=kpm_params)
        S_BA_i = -S_AB_i.H()
        assert not any((Y_AB.issparse(), S_AB_i.issparse(), S_BA_i.issparse()))
        S_AB.append(S_AB_i)
        S_BA.append(S_BA_i)
    S_AB = sum(S_AB)
    S_BA = -S_AB.H()

    # Generate effective Hamiltonian `Hd` to `order` order using `S`.
    # 0th commutator of H
    comm_j = ((H0_AA + H1_AA, H2_AB), (H2_BA, H0 + H1))
    # 0th order effective Hamiltonian
    Hd = H0_AA + H1_AA
    assert Hd == Hd.H(), Hd.todense()

    for j in range(1, order + 1):
        comm_j = _block_commute(comm_j, S_AB, S_BA)
        Hd += comm_j[0][0] * (1 / factorial(j))
        assert not Hd.issparse()
        assert Hd == Hd.H(), Hd.todense()

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
    if isinstance(a, scipy.sparse.spmatrix):
        return (a + b).A
    elif isinstance(b, scipy.sparse.spmatrix):
        return (b + a).A
    else:
        return a + b


class MatCoeffPolynomial(collections.defaultdict):

    # Make it work with numpy arrays
    __array_ufunc__ = None

    def __init__(self, *args, interesting_keys = None, **kwargs):
        super(MatCoeffPolynomial, self).__init__(lambda: 0, *args, **kwargs)
        self.interesting_keys = interesting_keys

    def copy(self):
        result = MatCoeffPolynomial()

        for key in self:
            try:
                result[key] = self[key].copy()
            except AttributeError:
                result[key] = self[key]

        result.interesting_keys = self.interesting_keys

        return result

    def clean_keys(self):
        """Removing all key that are not interesting."""
        if self.interesting_keys is not None:
            for key in list(self.keys()):
                if key not in self.interesting_keys:
                    del self[key]

    def tosympy(self, digits=12):
        """Convert MatCoeffPolynomial into sympy matrix."""
        result = [(key * np.round(val, digits)) for key, val in self.items()]
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
        """Lambdify MatCoeffPolynomial using gens as variables."""
        result = self.tosympy()
        result = sympy.lambdify(gens, result, 'numpy')
        return result

    def evalf(self, subs=None):
        result = []
        for key, val in self.items():
            key = float(key.evalf(subs=subs))
            result.append(key * val)
        return sum(result)

    def H(self):
        result = self.copy()

        for key, val in result.items():
            result[key] = val.T.conj()

        return result

    def __eq__(self, other):
        if not set(self) == set(other):
            return False
        a = self.todense()
        b = other.todense()
        for k, v in a.items():
            if not np.allclose(v, b[k]):
                return False
        return True

    def __neg__(self):
        result = self.copy()

        for key in result:
            result[key] *= -1

        return result

    def __add__(self, B):
        result = self.copy()

        try:
            for key in B:
                result[key] = _smart_add(result[key], B[key])
        except TypeError:
            result[1] = _smart_add(result[key], B)
        return result

    def __sub__(self, B):
        result = self.copy()

        try:
            for key in B:
                result[key] = _smart_add(result[key], -B[key])
        except TypeError:
            result[1] = _smart_add(result[key], -B)

        return result

    def __radd__(self, A):
        if (A == 0 or A == {}):
            return self.copy()
        else:
            return self + A

    def __rsub__(self, A):
        return -self + A

    def __mul__(self, other):
        # Multiplication by numbers, sympy symbols, arrays and Model
        if isinstance(other, Number):
            result = self.copy()
            for key, val in result.items():
                result[key] = other * val
        elif isinstance(other, Basic):
            result = MatCoeffPolynomial({key * other: val for key, val in self.items()})
            result.interesting_keys = self.interesting_keys
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(val, other)
        elif isinstance(other, MatCoeffPolynomial):
            result = MatCoeffPolynomial()
            for (k1, v1), (k2, v2) in product(self.items(), other.items()):
                result[k1 * k2] += _smart_dot(v1, v2)
            result.interesting_keys = list(set(self.interesting_keys) | set(other.interesting_keys))
            result.clean_keys()
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __truediv__(self, B):
        result = self.copy()

        if isinstance(B, MatCoeffPolynomial):
            raise TypeError(
                "unsupported operand type(s) for /: 'MatCoeffPolynomial' and "
                "'MatCoeffPolynomial'"
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
            result = MatCoeffPolynomial({other * key: val for key, val in self.items()})
            result.interesting_keys = self.interesting_keys
        elif isinstance(other, np.ndarray) or isinstance(other, scipy.sparse.spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = _smart_dot(other, val)
        else:
            raise NotImplementedError('Multiplication with type {} not implemented'.format(type(other)))
        return result

    def __pow__(self, n):
        if n == 0:
            return 1
        if n == 1:
            return self
        else:
            result = 1.0

        for i in range(n):
            result = result * self

        return result
