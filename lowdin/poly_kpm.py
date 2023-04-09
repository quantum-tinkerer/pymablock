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

import pdb

import numpy as np
import tinyarray as ta
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh
from lowdin.linalg import ComplementProjector, complement_projected
from lowdin.kpm_funcs import greens_function
from lowdin.misc import sym_to_ta, ta_to_symb
from lowdin.block_diagonalization import general, expanded
from lowdin.series import BlockSeries


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
        self.shape = (self.terms[0][0][0].shape[0], self.terms[-1][-1][0].shape[-1])

    def __add__(self, other):
        """
        Add two SumOfOperatorProducts objects.

        other : SumOfOperatorProducts

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(self.terms + other.terms)

    def __sub__(self, other):
        if isinstance(other, type(self)):
            return self + (-other)

    def __neg__(self):
        """
        Negate a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [
                [((-1) * sublist[0][0], sublist[0][1])] + sublist[1:]
                for sublist in self.terms
            ]
        )

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
        return SumOfOperatorProducts(
            [
                [(other * sublist[0][0], sublist[0][1])] + sublist[1:]
                for sublist in self.terms
            ]
        )

    def __rmul__(self, other):
        """
        Right multiply a SumOfOperatorProducts object by a scalar.

        other : scalar

        Returns:
        SumOfOperatorProducts
        """
        return self * other

    def adjoint(self):
        """
        Adjoint of a SumOfProductOperators object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [
                [
                    (v[0].conjugate().T, v[1][::-1])
                    if isinstance(v[0], np.ndarray)
                    else (v[0].adjoint(), v[1][::-1])
                    for v in reversed(slist)
                ]
                for slist in self.terms
            ]
        )

    def conjugate(self):
        """
        Conjugate a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [[(v[0].conjugate(), v[1]) for v in slist] for slist in self.terms]
        )

    def transpose(self):
        """
        Transpose a SumOfOperatorProducts object.

        Returns:
        SumOfOperatorProducts
        """
        return SumOfOperatorProducts(
            [
                [(v[0].transpose(), v[1][::-1]) for v in reversed(slist)]
                for slist in self.terms
            ]
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
        # check flags are equal
        temp = [
            self.reduce_sublist(slist, flag=["AA", "BA", "AB", "BB"])
            for slist in self.terms
        ]
        return reduce(add, (term[0][0] for term in temp))

    def flag(self):
        return self.evalf()[0][1]


def create_div_energs(
    h_0, vecs_a, eigs_a, vecs_b=None, eigs_b=None, kpm_params=None, precalculate_moments=False
):
    """
    h_0        : np.ndarray/ scipy.sparse.coo_matrix
                Rest hamiltonian of the system

    eigs_a     : np.ndarray
                Eigenvalues of the A subspace

    vecs_a     : np.ndarray
                Eigenvectors of the A subspace

    eigs_b     : np.ndarray
                (Sub)-Set of the eigenvalues of the B subspace

    vecs_b     : np.ndarray
                (Sub)-Set of the eigenvectors of the B subspace

    kpm_options: kpm_params and precalculate moments
                 as specified in kpm_fucs.

    Returns:
    divide_by_energies: callable
                        Function that applies divide by energies to the RHS of the Sylvester equation.
    """
    
    """
    MAke this interface for low-level Lowdin
    h is dict with symbols and sparse arrays
    does separation into aa,ab,bb
    and initialize lowdin classes
    target format is the {ta.array,something}
    
    """
    
    if vecs_b is None:
        vecs_b = np.empty((vecs_a.shape[0], 0))
    if eigs_b is None and vecs_b is not None:
        eigs_b = np.diag(vecs_b.conj().T @ h_0 @ vecs_b)
    if kpm_params is None:
        kpm_params = dict()

    def divide_by_energies(Y):
        #pdb.set_trace()
        Y = Y.to_array()
        if len(eigs_a) + len(eigs_b) < h_0.shape[0]:
            # KPM section
            Y_KPM = Y @ ComplementProjector(np.concatenate((vecs_a, vecs_b), axis=-1))

            vec_G_Y = greens_function(
                h_0,
                params=None,
                vectors=Y_KPM.conj(),
                kpm_params=kpm_params,
                precalculate_moments=precalculate_moments,
            )(eigs_a)
            res = np.vstack([vec_G_Y.conj()[:, m, m] for m in range(len(eigs_a))])

        else:
            # standard section
            res = np.zeros(Y.shape, dtype=complex)
        #pdb.set_trace()
        Y_ml = (Y @ ComplementProjector(vecs_a)) @ vecs_b
        G_ml = 1 / (np.array([eigs_a]).reshape(-1, 1) - eigs_b)
        res += (Y_ml * G_ml)@vecs_b.conjugate().transpose()
        return SumOfOperatorProducts([[(res, 'AB')]])

    return divide_by_energies


def numerical(h: dict, 
              vecs_a: np.ndarray, 
              eigs_a: np.ndarray, 
              vecs_b: np.ndarray = None, 
              eigs_b: np.ndarray = None, 
              kpm_params: dict = None, 
              precalculate_moments: bool = False):
    """
    assume h is dict with {1 : h_0, p_0 : h_p_0, ...}
    """
    hn, key_map = sym_to_ta(h) 
    n_infinite = len(key_map) 
    zero_index = ta.zeros(n_infinite)
    n = hn[ta.zeros(n_infinite)].shape[0]
    n_a = vecs_a.shape[-1] 
    
    p_b = ComplementProjector(vecs_a)
    
    # h_0
    
    """
    The separation needs to be changed to accomodate the situation when the spectrum not not ordered or the relevant subspace
    is not directly following each other
    """
    
    H_0_AA = SumOfOperatorProducts([[(np.diag(eigs_a), 'AA')]]) 
    H_0_BB = SumOfOperatorProducts([[(complement_projected(hn[zero_index], vecs_a), 'BB')]]) 
    
    # h_p
    H_p_AA = {k: SumOfOperatorProducts([[((vecs_a.conj().T @ v @ vecs_a) , 'AA')]]) for k,v in hn.items() if k != zero_index}
    H_p_BB = {k: SumOfOperatorProducts([[(complement_projected(v, vecs_a) , 'BB')]]) for k,v in hn.items() if k != zero_index}
    H_p_AB = {k: SumOfOperatorProducts([[((vecs_a.conj().T @ v @ p_b), 'AB')]]) for k,v in hn.items() if k != zero_index}

    H = BlockSeries(
        data={
            **{(0, 0) + zero_index: H_0_AA},
            **{(1, 1) + zero_index: H_0_BB},
            **{(0, 0) + tuple(key): value for key, value in H_p_AA.items()},
            **{(0, 1) + tuple(key): value for key, value in H_p_AB.items()},
            **{(1, 0) + tuple(key): value.conjugate().transpose() for key, value in H_p_AB.items()},
            **{(1, 1) + tuple(key): value for key, value in H_p_BB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )

    div_energs = create_div_energs(hn[zero_index], 
                                    vecs_a, 
                                    eigs_a, 
                                    vecs_b, 
                                    eigs_b, 
                                    kpm_params, 
                                    precalculate_moments)

    # use general algo
    h_tilde, u, u_ad = general(H, 
                               solve_sylvester = div_energs)

    #postprocessing
    """
    def h_tilde_eval(*index):
        if index[:2] in [[0,0],[0,1],[1,0]]:
            return h_tilde.evaluated[index].to_array()
        else:
            return h_tilde.evaluated[index]

    def u_eval(*index):
        if index[:2] in [[0,0],[0,1],[1,0]]:
            return u.evaluated[index].to_array()
        else:
            return u.evaluated[index]

    h_t_return = BlockSeries(eval=h_tilde_eval, shape=(2, 2), n_infinite=H.n_infinite)
    u_return = BlockSeries(eval = u_eval, shape=(2, 2), n_infinite=H.n_infinite)
    u_adj_return = BlockSeries(eval = lambda index: u_eval(*index).conjugate().transpose(), shape=(2, 2), n_infinite=H.n_infinite)
    """
    return h_tilde, u, u_ad




