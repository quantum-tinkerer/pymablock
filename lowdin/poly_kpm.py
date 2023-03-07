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
from scipy.linalg import eigh
from lowdin.linalg import ComplementProjector, complement_projected
from lowdin.kpm_funcs import greens_function


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


class get_bb_action(LinearOperator):
    """
    Need adjoint() that can distinguish the content of the arrays.
    """

    def __init__(self, op, vec_A):
        self.shape = op.shape
        self.op = op
        self.vec_A = vec_A
        self.dtype = complex

    def _matvec(self, v):
        temp = self.op @ (v - self.vec_A @ (self.vec_A.conj().T @ v))
        return temp - self.vec_A @ (self.vec_A.conj().T @ temp)

    def _matmat(self, V):
        temp = self.op @ (V - self.vec_A @ (self.vec_A.conj().T @ V))
        return temp - self.vec_A @ (self.vec_A.conj().T @ temp)

    def _rmatvec(self, v):
        temp = (v - (v @ self.vec_A) @ self.vec_A.conj().T) @ self.op
        return temp - (temp @ self.vec_A) @ self.vec_A.conj().T

    def _rmatmat(self, V):
        temp = (V - (V @ self.vec_A) @ self.vec_A.conj().T) @ self.op
        return temp - (temp @ self.vec_A) @ self.vec_A.conj().T

    def _adjoint(self):
        return get_bb_action(self.op, self.vec_A)

    def conjugate(self):
        return get_bb_action(self.op, self.vec_A)

    def __rmatmul__(self, other):
        try:
            res = self._rmatmat(other)
        except:
            res = self._matvec(other)
        return res

    def __truediv__(self, other):
        return 1 / other * self

    __array_ufunc__ = None


def create_div_energs_old(e_a, v_a, H_0_BB):
    if isinstance(H_0_BB, get_bb_action):
        H_0_BB = H_0_BB @ np.eye(H_0_BB.shape[0])
    if isinstance(H_0_BB,SumOfOperatorProducts):
        H_0_BB = H_0_BB.to_array() @ np.eye(H_0_BB.to_array().shape[0])

    n_a = len(e_a)
    a_eigs = e_a

    b_eigs, b_vecs = eigh(H_0_BB @ np.eye(H_0_BB.shape[0]))
    # find out where the A-subspace is
    a_inds = np.argmax(
        np.abs(v_a.conj().T @ b_vecs), axis=-1
    )  # this needs uniqueness check
    b_eigs = np.delete(b_eigs, a_inds)
    b_vecs = np.delete(b_vecs, a_inds, axis=1)

    e_div = 1 / (a_eigs.reshape(-1, 1) - b_eigs)
    full_vecs = np.hstack((v_a, b_vecs))

    def divide_energies(Y):
        type_flag = isinstance(Y, SumOfOperatorProducts)

        if  type_flag:
            Y = Y.to_array()
            type_flag = True

        V = Y @ full_vecs
        V[:, n_a:] = V[:, n_a:] * e_div
        Y = V @ full_vecs.T.conj()
        
        if type_flag:
            Y = SumOfOperatorProducts([[(Y, "AB")]])
        
        return Y

    return divide_energies


def create_div_energs(
    h_0, vecs_a, eigs_a, vecs_b=None, eigs_b=None, kpm_params=dict(), precalculate_moments=False
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
        if len(eigs_a) + len(eigs_b) < h_0.shape[0]:
            # KPM section
            Y_KPM = (
                Y
                - Y.dot(vecs_a).dot(vecs_a.T.conj())
                - Y.dot(vecs_b).dot(vecs_b.T.conj())
            )
            # This way we do all vectors for all energies, uses a bit more RAM than
            # absolutely necessary, but makes the code simpler.
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

        Y_ml = Y.dot(vecs_b)
        G_ml = 1 / (np.array([eigs_a]).reshape(-1, 1) - eigs_b)
        res += (Y_ml * G_ml).dot(vecs_b.T.conj())
        return res

    return divide_by_energies


def divide_energies(Y, H_0_AA, H_0_BB, mode="arr"):
    """
    Divide the right hand side of the equation by the energy denominators.

    Y : np.array of shape (n, m) right hand side of the equation
    H_0_AA : SumOfOperatorProducts object
    H_0_BB : SumOfOperatorProducts object

    Returns:
    Y divided by the energy denominators
    """

    if mode == "arr":
        E_A = np.diag(H_0_AA.to_array())
        E_B = np.diag(H_0_BB.to_array())

    if mode == "op":
        if isinstance(H_0_AA, np.ndarray):
            E_A = np.diag(H_0_AA)
            E_B = np.diag(H_0_BB @ np.eye(H_0_BB.shape[0]))
        else:
            E_A = np.diag(H_0_AA.to_array())
            E_B = np.diag(H_0_BB.to_array() @ np.eye(H_0_BB.to_array().shape[0]))
        # E_B is secretly E_A+A_B
        # E_B = np.concatenate((E_A, E_B[np.where(E_B != 0)[0]]))
        E_B = E_B

    energy_denoms = 1 / (E_A.reshape(-1, 1) - E_B)
    energy_denoms[:, np.where(E_B == 0)[0]] = 0

    return Y * energy_denoms
