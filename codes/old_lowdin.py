from . import old_perturbationS as ps

import collections

from functools import reduce
from operator import mul
from itertools import product
from scipy.sparse import csr_matrix
from math import factorial
import sympy
import numpy as np


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


def divide_by_energies(rhs, all_energies, indices):
    indexes_B = [i for i in range(len(all_energies)) if i not in indices]

    output = MatCoeffPolynomial()
    for key, val in rhs.items():
        if type(val) == csr_matrix:
            val = val.toarray()

        for i,j in product(indices, indexes_B):
            val[i,j] = val[i,j] * 1.0 / (all_energies[i] - all_energies[j])

        for i,j in product(indexes_B, indices):
            val[i,j] = val[i,j] * 1.0 / (all_energies[i] - all_energies[j])

        output[key] = csr_matrix(val)
    return output


def precalculate_in_basis(polynomial, basis):
    """ Precalculate perturbation in given basis.

    polynomial: dictionary form of matrix polynomial
                or equivalent
    """
    basis = csr_matrix(basis)
    output = MatCoeffPolynomial()
    for key, val in polynomial.items():
        element = basis.transpose().conjugate() @ csr_matrix(val) @ basis
        output[key] = np.asarray(element.toarray())
    return output


def get_H0_H1_H2(all_energies, precalculated, indices):
    """ Return H0, H1, H2 in form of MatCoeffPolynomial. """

    indexes_B = [i for i in range(len(all_energies)) if i not in indices]

    H0 = MatCoeffPolynomial()
    H0[1] = np.diag(all_energies)

    H1 = MatCoeffPolynomial()
    for key in precalculated.keys():
        T = precalculated[key]
        tmp = np.zeros_like(T)

        for i,j in product(indices, indices):
            tmp[i,j] = T[i,j]

        for i,j in product(indexes_B, indexes_B):
            tmp[i,j] = T[i,j]

        H1[key] = tmp

    H2 = MatCoeffPolynomial()
    for key in precalculated.keys():
        T = precalculated[key]
        tmp = np.zeros_like(T)

        for i,j in product(indices, indexes_B):
            tmp[i,j] = T[i,j]

        for i,j in product(indexes_B, indices):
            tmp[i,j] = T[i,j]

        H2[key] = tmp

    return H0, H1, H2


def get_effective_model(ev, evec, indices, perturbation, interesting_keys=None, prec=12):
    """Return effective model for given perturbation.

    Implementation of quasi-degenerated perturbation theory.
    Inspired by appendix in R. Winkler book.

    input
    -----
    ev: energies
    evec: basis states
    indices: which states from evec-s are to be in group A

    perturbation: dictionary: keys are sympy symbols (parameters
        of perturbation) and values are corresponding matrix representations
        of perturbation hamiltonian. Basis in which perturbation is given
        should be the same as basis in which basis is calculated.
    """
    def polycommute(a, b):
        return polydot(a,b, interesting_keys) - polydot(b,a, interesting_keys)

    def polycommute_n(a,b,n):
        if n == 0:
            return a
        else:
            res = polycommute(a,b)
            for i in range(n-1):
                res = polycommute(res, b)
        return res

    if interesting_keys is None:
        interesting_keys = get_maximum_powers(perturbation.keys(), 2)

    precalculated = precalculate_in_basis(perturbation, evec)
    H0, H1, H2 = get_H0_H1_H2(ev, precalculated, indices)

    def transform_to_sparse(polynomial):
        output = MatCoeffPolynomial()
        for key, val in polynomial.items():
            output[key] = csr_matrix(val)

        output.interesting_keys = polynomial.interesting_keys
        output.remove_not_interesting_keys(interesting_keys)
        return output

    H0.interesting_keys = interesting_keys
    H1.interesting_keys = interesting_keys
    H2.interesting_keys = interesting_keys

    H0 = transform_to_sparse(H0)
    H1 = transform_to_sparse(H1)
    H2 = transform_to_sparse(H2)

    rhs = ps.Y1
    rhs = rhs(H0, H1, H2)
    S1 = divide_by_energies(rhs, ev, indices)
    S1.interesting_keys = interesting_keys

    rhs = ps.Y2
    rhs = rhs(H0, H1, H2, S1)
    S2 = divide_by_energies(rhs, ev, indices)
    S2.interesting_keys = interesting_keys

    rhs = ps.Y3
    rhs = rhs(H0, H1, H2, S1, S2)
    S3 = divide_by_energies(rhs, ev, indices)
    S3.interesting_keys = interesting_keys

    rhs = ps.Y4
    rhs = rhs(H0, H1, H2, S1, S2, S3)
    S4 = divide_by_energies(rhs, ev, indices)
    S4.interesting_keys = interesting_keys

    rhs = ps.Y5
    rhs = rhs(H0, H1, H2, S1, S2, S3, S4)
    S5 = divide_by_energies(rhs, ev, indices)
    S5.interesting_keys = interesting_keys

    S = S1 + S2 + S3 + S4 + S5
    S.interesting_keys = interesting_keys

    S = transform_to_sparse(S)

    Hd = polycommute_n(H0+H1, S, 0) + polycommute_n(H2, S, 1)

    jmax = 3
    for j in range(1, jmax+1):
        Hd += polycommute_n(H0+H1, S, 2*j) * (1.0 / factorial(2*j))
        Hd += polycommute_n(H2, S, 2*j+1)  * (1.0 / factorial(2*j + 1))

    for key, val in Hd.items():
        val = val.toarray()[:, list(indices)][list(indices), :]
        Hd[key] = np.round(val, prec)

    return Hd


# *********************** POLYNOMIAL CLASS ************************************
class MatCoeffPolynomial(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super(MatCoeffPolynomial, self).__init__(lambda: 0, *args, **kwargs)
        self.interesting_keys = None

    def copy(self):
        result = MatCoeffPolynomial()

        for key in self:
            try:
                result[key] = self[key].copy()
            except AttributeError:
                result[key] = self[key]

        result.interesting_keys = self.interesting_keys

        return result

    def remove_not_interesting_keys(self, interesting_keys):
        """Removing all key that are not interesting."""
        for key in list(self.keys()):
            if key not in interesting_keys:
                del self[key]

    def tosympy(self):
        """Convert MatCoeffPolynomial into sympy matrix."""
        result = [(key * val) for key, val in self.items()]
        result = sympy.Matrix(sum(result))
        result.simplify()
        return result

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
        return self + A

    def __rsub__(self, A):
        return -self + A

    def __mul__(self, B):
        result = self.copy()

        if isinstance(B, MatCoeffPolynomial):
            interesting_keys = self.interesting_keys
            if B.interesting_keys != None:
                interesting_keys = list(np.concatenate([interesting_keys] + [B.interesting_keys]))

            result = polydot(self, B, interesting_keys)
            result.interesting_keys = interesting_keys
        else:
            for key in result:
                result[key] *= B

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

    def __rmul__(self, A):
        # __mul__ implements *, which is interpreted as elementwise multiplication
        # this is commutative!
        return self * A

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


def polydot(A, B, interesting_keys=None):
    result = MatCoeffPolynomial()
    for keyA, valA in A.items():
        for keyB, valB in B.items():

            out_key = keyA*keyB
            if interesting_keys != None and out_key not in interesting_keys:
                pass
            else:
                result[out_key] += valA.dot(valB)

    return result