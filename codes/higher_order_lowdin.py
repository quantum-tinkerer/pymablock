from .perturbationS import Y_i

import collections

from functools import reduce
from operator import mul
from itertools import product
from scipy.sparse import csr_matrix, spmatrix
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


def divide_by_energies(Y_AB, energies_A, vectors_A, H_0, kpm_params):
    S_AB = MatCoeffPolynomial()
    S_AB.interesting_keys = Y_AB.interesting_keys
    if isinstance(vectors_A, spmatrix):
        vectors_A = vectors_A.A
    for key, val in Y_AB.items():
        if isinstance(val, spmatrix):
            val = val.toarray()

        res = []
        for m, E_m in enumerate(energies_A):
            G_Y_vec_m = build_greens_function(H_0, params=None, vectors=val[m].conj(), kpm_params=kpm_params)
            res.append(G_Y_vec_m(E_m).conj())
        res = np.vstack(res)

        S_AB[key] = res - res.dot(vectors_A.T.conj()).dot(vectors_A)
    # S_AB = vectors_A * (vectors_A.T.conj() * S_AB)
    return S_AB.tosparse()


def commute_n_even(H_AA, H_BB, S_AB, S_BA, n):
    res_AA = H_AA.copy()
    res_BB = H_BB.copy()
    for i in range(n):
        res_AB = res_AA * S_AB - S_AB * res_BB
        res_BA = res_BB * S_BA - S_BA * res_AA
        res_AA = res_AB * S_BA - S_AB * res_BA
        res_BB = res_BA * S_AB - S_BA * res_AB
    return res_AA


def commute_n_odd(H_AB, H_BA, S_AB, S_BA, n):
    res_AA = H_AB * S_BA - S_AB * H_BA
    res_BB = H_BA * S_AB - S_BA * H_AB
    for i in range(n):
        res_AB = res_AA * S_AB - S_AB * res_BB
        res_BA = res_BB * S_BA - S_BA * res_AA
        res_AA = res_AB * S_BA - S_AB * res_BA
        res_BB = res_BA * S_AB - S_BA * res_AB
    return res_AA


def get_effective_model(H0, H1, evec_A, interesting_keys=None, order=2, kpm_params=None):
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
        as a set of row vectors
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

    N = H0.shape[0]
    H0 = MatCoeffPolynomial({1: H0}, interesting_keys = interesting_keys)
    H0 = H0.tosparse()
    H1 = MatCoeffPolynomial(H1, interesting_keys = interesting_keys)
    H1 = H1.tosparse()
    evec_A = csr_matrix(evec_A)

    H0_AA = evec_A * H0 * evec_A.T.conj()
    assert H0_AA == H0_AA.H()
    ev_A = np.diag(H0_AA.todense()[1])
    assert np.allclose(np.diag(ev_A), H0_AA.todense()[1]), 'evec_A should be eigenvectors of H0'
    H1_AA = evec_A * H1 * evec_A.T.conj()
    assert H1_AA == H1_AA.H()
    H2_AB = evec_A * H1 - H1_AA * evec_A
    H2_BA = H1 * evec_A.T.conj() - evec_A.T.conj() * H1_AA
    assert H2_AB == H2_BA.H()

    H0 = H0.tosparse()
    H0_AA = H0_AA.tosparse()
    assert H0_AA == H0_AA.H(), H0_AA.todense()
    H1 = H1.tosparse()
    H1_AA = H1_AA.tosparse()
    H2_AB = H2_AB.tosparse()
    H2_BA = H2_BA.tosparse()

    S_AB = []
    S_BA = []
    for i in range(1, order):
        Y = Y_i[i - 1]
        Y_AB = Y(H0_AA, H0, H1_AA, H1, H2_AB, H2_BA, S_AB, S_BA)
        # print('Y_AB', Y_AB.todense())
        S_AB_i = divide_by_energies(Y_AB, ev_A, evec_A, H0[1], kpm_params=kpm_params)
        # print('S_AB_i', S_AB_i.todense())
        # print('step1')
        S_BA_i = -S_AB_i.H()
        S_AB.append(S_AB_i)
        S_BA.append(S_BA_i)
        # print('step2')

    S_AB = sum(S_AB)
    # print('S_AB', S_AB.todense())
    S_BA = -S_AB.H()

    # print('step4')
    assert H0_AA == H0_AA.H()
    assert H1_AA == H1_AA.H()
    assert H2_AB == H2_BA.H()
    assert S_AB == -S_BA.H()
    Hd = H0_AA + H1_AA + H2_AB * S_BA - S_AB * H2_BA
    # print('Hd', Hd.todense())
    assert Hd == Hd.H(), Hd.todense()

    # print('step5')
    for j in range(1, order//2 + 1):
        # print(j)
        Hd += commute_n_even(H0_AA + H1_AA, H0 + H1, S_AB, S_BA, j) * (1 / factorial(2*j))
        # print('Hd', Hd.todense())
        Hd += commute_n_odd(H2_AB, H2_BA, S_AB, S_BA, j) * (1 / factorial(2*j + 1))
        # print('Hd', Hd.todense())
        assert Hd == Hd.H(), Hd.todense()

    Hd = Hd.todense()

    return Hd


# *********************** POLYNOMIAL CLASS ************************************
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
            output[key] = csr_matrix(val)
        return output

    def todense(self):
        output = self.copy()
        for key, val in output.items():
            if isinstance(val, spmatrix):
                output[key] = val.toarray()
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
                result[key] += B[key]
        except TypeError:
            result[1] += B

        return result

    def __sub__(self, B):
        result = self.copy()

        try:
            for key in B:
                result[key] -= B[key]
        except TypeError:
            result[1] -= B

        return result

    def __radd__(self, A):
        if not A:
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
        elif isinstance(other, np.ndarray) or isinstance(other, spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = np.dot(val, other)
            # result.shape = np.dot(np.zeros(self.shape), other).shape
        elif isinstance(other, MatCoeffPolynomial):
            result = MatCoeffPolynomial()
            for (k1, v1), (k2, v2) in product(self.items(), other.items()):
                result[k1 * k2] += v1.dot(v2)
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
        elif isinstance(other, np.ndarray) or isinstance(other, spmatrix):
            result = self.copy()
            for key, val in list(result.items()):
                result[key] = other.dot(val)
            # result.shape = np.dot(other, np.zeros(self.shape)).shape
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
