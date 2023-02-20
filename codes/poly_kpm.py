# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from itertools import product
from functools import reduce
from operator import add

import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import block_diag, eigh
from numpy.linalg import multi_dot


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
        summed_value = reduce(add, (term[0][0] for term in length1_terms))
        label = length1_terms[0][0][1]
        self.terms = new_terms + [[(summed_value, label)]]
    
    def to_array(self):
        #check flags are equal
        temp = [self.reduce_sublist(slist,flag=['AA','BA','AB','BB']) for slist in self.terms]
        return reduce(add,(term[0][0] for term in temp))
    
    def flag(self):
        return self.evalf()[0][1]

def create_div_energs(H_0_AA, H_0_BB, mode='arr'):

    if mode=='op':
        H_A = H_0_AA.to_array()
        n_a = H_A.shape[0]
        H_B = (H_0_BB.to_array() @ np.eye(H_0_BB.to_array().shape[0]))[n_a:,n_a:]
    else:
        H_A = H_0_AA
        n_a = H_0_AA.shape[0]
        H_B = H_0_BB

    H_0 = block_diag(H_A, H_B)
    
    e_a = eigh(H_A,eigvals_only=True)
    e_b = eigh(H_B,eigvals_only=True)
    
    all_eig = np.concatenate((e_a, e_b))
    eigs, vecs = eigh(H_0)
    order = np.concatenate([np.where(np.isclose(all_eig[i],eigs))[0]
                            for i in range(len(all_eig))
                            if not np.any(all_eig[i]==all_eig[:i])])
    eigs, vecs = eigs[order], vecs[:,order]
    assert np.allclose(all_eig, eigs)
    
    e_div = 1/(all_eig.reshape(-1,1)-all_eig)
    for i in range(len(all_eig)):
        e_div[i,i] = 0
    e_div = vecs.conj().T @ e_div @ vecs
    e_div[:,:n_a] = 0
    
    def divide_energies(Y):
        return Y * e_div[:n_a,:]
        
    return divide_energies

def divide_energies(Y, H_0_AA, H_0_BB, mode='arr'):
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
    
    if mode == 'arr':
        E_A = np.diag(H_0_AA.to_array())
        E_B = np.diag(H_0_BB.to_array())
    
    if mode == 'op':
        E_A = np.diag(H_0_AA.to_array())
        E_B = np.diag(H_0_BB.to_array() @ np.eye(H_0_BB.to_array().shape[0]))
        # E_B is secretly E_A+A_B
        #E_B = np.concatenate((E_A, E_B[np.where(E_B != 0)[0]]))
        E_B = E_B

    energy_denoms = 1 / (E_A.reshape(-1, 1) - E_B)
    energy_denoms[:,np.where(E_B == 0)[0]] = 0

    return Y * energy_denoms


class get_bb_action(LinearOperator):
    def __init__(self, op, vec_A):
       self.shape = op.shape
       self.op = op
       self.vec_A = vec_A
       self.dtype = complex
       
    def _matvec(self, v):
        temp = self.op @ (v - self.vec_A @ (self.vec_A.conj().T @ v) )
        return (temp - self.vec_A @ (self.vec_A.conj().T @ temp))
    
    def _matmat(self, V):
        temp = self.op @ (V - self.vec_A @ (self.vec_A.conj().T @ V))
        return (temp - self.vec_A @ (self.vec_A.conj().T @ temp))
    
    def _rmatvec(self, v):
        temp = (v - (v @ self.vec_A) @ self.vec_A.conj().T) @ self.op
        return (temp - (temp @ self.vec_A) @ self.vec_A.conj().T)
    
    def _rmatmat(self, V):
        temp = (V - (V @ self.vec_A) @ self.vec_A.conj().T) @ self.op
        return (temp - (temp @ self.vec_A) @ self.vec_A.conj().T)

    def _adjoint(self):
        return get_bb_action(self.op, self.vec_A)
    
    def __rmatmul__(self,other):
        try: 
            res = self._rmatmat(other)
        except:
            res = self._matvec(other)
        return res
    
    __array_ufunc__ = None

        

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
    [(rnd((10, 4)),'BA'), (rnd((4, 10)), "AB"), (rnd((10, 10)), "BB")],
    [(rnd((10, 4)), "BA"), (rnd((4, 4)), "AA"), (rnd((4, 10)), "AB")],
]

t_list_3 = [[(rnd((10, 4)), "BA"), (rnd((4, 10)), "AB")], [(rnd((4, 4)), "AA")]]