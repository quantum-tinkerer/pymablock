from itertools import product
from functools import reduce

import numpy as np
from scipy.sparse.linalg import LinearOperator


class SumOfOperatorProducts:
    """
    A class that represents a sum of operator products.
    """

    def __init__(self, terms):
        """
        Initialize a SumOfOperatorProducts object.

        terms : list of lists of tuples (array, str)
            The first element of the tuple is the operator, the second is a string
            "AA", "AB", "BA", "BB" that specifies the subspaces it couples.
        """
        # All terms must be nonempty, and should couple the same subspaces.
        if not all(terms):
            raise ValueError("Empty term.")
        starts = (term[0][1][0] for term in terms)
        ends = (term[-1][1][1] for term in terms)
        if len(set(starts)) > 1 or len(set(ends)) > 1:
            raise ValueError("Terms couple different subspaces.")
        # All operators should couple compatible spaces.
        for term in terms:
            for op1, op2 in zip(term, term[1:]):
                if op1[1][1] != op2[1][0]:
                    raise ValueError("Terms couple incompatible subspaces.")

        self.terms = terms
        self.simplify_products()

    def __add__(self, other):
        """
        Add two SumOfOperatorProducts objects.

        other : SumOfOperatorProducts

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(self.terms + other.terms)
    
    def __sub__(self,other):
        if isinstance(other,type(self)):
            return self + (-other)
        

    def __neg__(self):
        """
        Negate a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts([[((-1) * sublist[0][0], sublist[0][1])] + sublist[1:] for sublist in self.terms])

    def __matmul__(self, other):
        """
        Multiply two SumOfOperatorProducts objects.

        other : SumOfOperatorProducts

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [a + b for a, b in product(self.terms, other.terms)]
        )

    def __truediv__(self, other):
        """
        Divide a SumOfOperatorProducts object by a scalar.

        other : scalar

        Returns:
        SumOfOperatorProducts
        """
        return (1 / other) * self

    def __mul__(self, other):
        """
        Multiply a SumOfOperatorProducts object by a scalar.

        other : scalar

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts([[(other * sublist[0][0], sublist[0][1])]+sublist[1:] for sublist in self.terms])

    def __rmul__(self, other):
        """
        Right multiply a SumOfOperatorProducts object by a scalar.

        other : scalar

        Returns:
        SumOfOperatorProducts
        """
        return self * other

    def conjugate(self):
        """
        Conjugate a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts([[(v[0].conjugate(), v[1]) for v in slist] for slist in self.terms])

    def transpose(self):
        """
        Transpose a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [[(v[0].transpose(), v[1][::-1]) for v in reversed(slist)]
             for slist in self.terms]
        )

    def reduce_sublist(self, slist, flag=["AA", "AB", "BA"]):
        """
        Reduce a sublist of a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        # This can be made more efficient by getting rid of the surplus loop
        # to check equality
        def elmmul(a, b):
            return (a[0] @ b[0], "{0}{1}".format(a[1][0], b[1][1]))

        temp = [slist[0]]
        for v in slist[1:]:
            if (str(temp[-1][1][0]) + str(v[1][1])) in flag:
                temp[-1] = elmmul(temp[-1], v)
            else:
                temp.append(v)
        if len(temp) < len(slist):
            return self.reduce_sublist(temp)
        elif len(temp) == len(slist):
            return temp

    def simplify_products(self):
        """
        Simplify a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        self.terms = [self.reduce_sublist(slist) for slist in self.terms]
        self._add_explicit_terms()

    def _add_explicit_terms(self):
        """Sum all terms of length 1 inplace

        Because every term of length 1 is a single operator, we can sum them directly
        """
        terms = self.terms
        new_terms = []
        length1_terms = []
        for term in terms:
            if len(term) == 1:
                length1_terms.append(term)
            else:
                new_terms.append(term)
        if not length1_terms:
            return
        summed_value = sum(term[0][0] for term in length1_terms)
        label = length1_terms[0][0][1]
        self.terms = new_terms + [[(summed_value, label)]]
    
    def to_array(self):
        #check flags are equal
        temp = [self.reduce_sublist(slist,flag=['AA','BA','AB','BB']) for slist in self.terms]
        return np.array(sum([term[0][0] for term in temp]))
    
    def flag(self):
        return self.evalf()[0][1]


def divide_energies(Y, H_0_AA, H_0_BB):
    """
    Divide the right hand side of the equation by the energy denominators.

    Y : np.array of shape (n, m) right hand side of the equation
    H_0_AA : SumOfOperatorProducts object
    H_0_BB : SumOfOperatorProducts object

    Returns:
    Y divided by the energy denominators
    """
    
    assert len(Y.terms) == 1
    assert len(Y.terms[0]) ==1
    
    E_A = np.diag(H_0_AA.to_array())
    E_B = np.diag(H_0_BB.to_array())
    energy_denoms = 1 / (E_A.reshape(-1, 1) - E_B)

    return Y * energy_denoms


# +
class h_0_op(LinearOperator):
    def __init__(self, h_0):
        self.h_0 = h_0
        self.shape = h_0.shape
        self.dtype = complex

    def _matvec(self,v):
        return np.dot(self.h_0, v)

    def _matmat(self,V):
        return np.dot(self.h_0, V)

    def _adjoint(self):
        return h_0_op(self.h_0)

class p_b_op(LinearOperator):
    def __init__(self,vec_A):
        self.vec_A = vec_A
        self.shape = (vec_A.shape[0],vec_A.shape[0])
        self.dtype = complex
        
    def _matvec(self,v):
        p = self.vec_A.conj().T @ self.vec_A
        v1 = v[:p.shape[-1]] - p @ v[:p.shape[-1]]
        v2 = v[p.shape[-1]:]
        return np.concatenate((v1,v2),axis=0)
        
    def _adjoint(self):
        return p_b(self.vec_A)


# +
from numpy.random import random as rnd

t_list = [
    [
        (rnd((4, 10)), "AB"),
        (rnd((10, 10)), "BB"),
        (rnd((10, 4)), "BA"),
        (rnd((4, 10)), "AB"),
    ],
    [(rnd((4, 10)), "AB"), (rnd((10, 10)), "BB")],
    [
        (rnd((4, 4)), "AA"),
        (rnd((4, 10)), "AB"),
        (rnd((10, 4)), "BA"),
        (rnd((4, 10)), "AB"),
        (rnd((10, 10)), "BB"),
        (rnd((10, 4)), "BA"),
        (rnd((4, 4)), "AA"),
        (rnd((4, 10)), "AB"),
    ],
]

t_list_2 = [
    [(rnd((4, 10)), "AB"), (rnd((10, 10)), "BB")],
    [(rnd((10, 4)), "BA"), (rnd((4, 4)), "AA"), (rnd((4, 10)), "AB")],
]

t_list_3 = [[(rnd((10, 4)), "BA"), (rnd((4, 10)), "AB")], [(rnd((4, 4)), "AA")]]
