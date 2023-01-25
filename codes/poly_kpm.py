from itertools import product
from functools import reduce

import numpy as np


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
        ends = (term[0][1][1] for term in terms)
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

    def __neg__(self):
        """
        Negate a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        temp = []
        for sublist in self.terms:
            sublist[0] = ((-1) * sublist[0][0], sublist[0][1])
            temp.append(sublist)
        return SumOfOperatorProducts(temp)

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
        temp = []
        for sublist in self.terms:
            sublist[0] = (other * sublist[0][0], sublist[0][1])
            temp.append(sublist)
        return SumOfOperatorProducts(temp)

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
        temp = [[(v[0].conjugate(), v[1]) for v in slist] for slist in self.terms]
        return SumOfOperatorProducts(temp)

    def transpose(self):
        """
        Transpose a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        temp = [[(v[0].transpose(), v[1]) for v in slist] for slist in self.terms]
        return SumOfOperatorProducts(temp)

    def reduce_sublist(self, slist):
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
            if (str(temp[-1][1][0]) + str(v[1][1])) in ["AA", "AB", "BA"]:
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
        nterms = [self.reduce_sublist(slist) for slist in self.terms]
        self.terms = nterms
        self._add_explicit_terms()

    def sum_sublist(self, slist, flag):
        """
        Sum a sublist of a SumOfOperatorProducts object.

        slist : list of tuples with (array, str)
        flag : str (AA, AB, BA, BB)

        Returns:
        SumOfOperatorProducts
        """
        return sum([v[0] for v in slist if v[1] == flag])

    def evalf(self, flag=None):
        """
        Evaluate a SumOfOperatorProducts object.

        flag : str (AA, AB, BA, BB)

        Returns:
        SumOfOperatorProducts
        """
        temp = [self.reduce_sublist(slist) for slist in self.terms]

        sec_temp = []
        if flag is not None:
            sec_temp.append(
                [v for v in temp if (str(v[0][1][0]) + str(v[-1][1][1])) == flag]
            )
        else:
            sec_temp = temp
        return sec_temp

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


def divide_energies(Y, H_0_AA, H_0_BB):
    """
    Divide the right hand side of the equation by the energy denominators.

    Y : np.array of shape (n, m) right hand side of the equation
    H_0_AA : SumOfOperatorProducts object
    H_0_BB : SumOfOperatorProducts object

    Returns:
    Y divided by the energy denominators
    """
    E_A = np.diag(H_0_AA.evalf("AA")[0][0])
    E_B = np.diag(H_0_BB.evalf("BB")[0][0])
    energy_denoms = 1 / (E_A.reshape(-1, 1) - E_B)

    return Y * energy_denoms


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
