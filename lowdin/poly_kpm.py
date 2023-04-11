from itertools import product
from functools import reduce
from operator import add

import numpy as np

from lowdin.linalg import ComplementProjector, complement_projected
from lowdin.kpm_funcs import greens_function
from lowdin.block_diagonalization import general
from lowdin.series import BlockSeries, zero

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
    h_0,
    vecs_a,
    eigs_a,
    vecs_b=None,
    eigs_b=None,
    kpm_params=None,
    precalculate_moments=False,
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
    if vecs_b is None:
        vecs_b = np.empty((vecs_a.shape[0], 0))
    if eigs_b is None:
        eigs_b = np.diag(vecs_b.conj().T @ h_0 @ vecs_b)
    if kpm_params is None:
        kpm_params = dict()

    need_kpm = len(eigs_a) + len(eigs_b) < h_0.shape[0]
    need_explicit = bool(len(eigs_b))
    if not any((need_kpm, need_explicit)):
        # B subspace is empty
        return lambda Y: Y

    if need_kpm:
        kpm_projector = ComplementProjector(np.concatenate((vecs_a, vecs_b), axis=-1))

        def sylvester_kpm(Y):
            Y_KPM = Y @ kpm_projector
            vec_G_Y = greens_function(
                h_0,
                params=None,
                vectors=Y_KPM.conj(),
                kpm_params=kpm_params,
                precalculate_moments=precalculate_moments,
            )(eigs_a)
            return np.vstack([vec_G_Y.conj()[:, m, m] for m in range(len(eigs_a))])

    if need_explicit:
        G_ml = 1 / (eigs_a[:, None] - eigs_b[None, :])

        def sylvester_explicit(Y):
            return ((Y @ vecs_b) * G_ml) @ vecs_b.conj().T

    def solve_sylvester(Y):
        Y = Y.to_array()
        if need_kpm and need_explicit:
            result = sylvester_kpm(Y) + sylvester_explicit(Y)
        elif need_kpm:
            result = sylvester_kpm(Y)
        elif need_explicit:
            result = sylvester_explicit(Y)

        return SumOfOperatorProducts([[(result, "AB")]])

    return solve_sylvester


def numerical(
    h: dict,
    vecs_a: np.ndarray,
    eigs_a: np.ndarray,
    vecs_b: np.ndarray = None,
    eigs_b: np.ndarray = None,
    kpm_params: dict = None,
    precalculate_moments: bool = False,
):
    """

    Parameters:
    -----------
    h : dict of np.ndarray or scipy.sparse matrices
        Full Hamiltonian of the system with keys corresponding to the order of
        the perturbation series
    vecs_a : np.ndarray or scipy.sparse matrix
        eigenvectors of the A (effective) subspace of H_0 (h[1])
    eigs_a : np.ndarray
        eigenvalues to the aforementioned eigenvectors
    vecs_b : np.ndarray or scipy.sparse matrix, optional
        Explicit parts of the B (auxilliary) space. Need to be eigenvectors
        to H_0 (h[1])
    eigs_b : np.ndarray, optional
        eigenvectors to the aforementioned explicit B space eigenvectors
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    precalculate_moments: bool, default False
        Whether to precalculate and store all the KPM moments of `vectors`.
        This is useful if the Green's function is evaluated at a large
        number of energies, but uses a large amount of memory.
        If False, the KPM expansion is performed every time the Green's
        function is called, which minimizes memory use.


    Returns:
    --------
    h_t : BlockSeries
        Full block diagonalized Hamiltonian of the problem. The explict
        entries are called via var_name.evaluated[x, y, order] where
        x, y refer to the sub-space index (A, B).
        E.g.: var_name.evaluated[0,1,3] returns the AB entry of
        var_name to third order in a single perturbation parameter.
    u : BlockSeries
        Unitary transformation that block diagonalizes the initial perturbed
        Hamiltonian. Entries are accessed the same way as h_t
    u_adj : BlockSeries
        Adjoint of u.
    """
    n_infinite = len(next(iter(h)))
    zero_index = (0,) * n_infinite

    p_b = ComplementProjector(vecs_a)

    H_0_AA = SumOfOperatorProducts([[(np.diag(eigs_a), "AA")]])
    H_0_BB = SumOfOperatorProducts(
        [[(complement_projected(h[zero_index], vecs_a), "BB")]]
    )

    H_p_AA = {
        k: SumOfOperatorProducts([[((vecs_a.conj().T @ v @ vecs_a), "AA")]])
        for k, v in h.items()
        if any(k)
    }
    H_p_BB = {
        k: SumOfOperatorProducts([[(complement_projected(v, vecs_a), "BB")]])
        for k, v in h.items()
        if any(k)
    }
    H_p_AB = {
        k: SumOfOperatorProducts([[((vecs_a.conj().T @ v @ p_b), "AB")]])
        for k, v in h.items()
        if any(k)
    }

    H = BlockSeries(
        data={
            (0, 0) + n_infinite * (0,): H_0_AA,
            (1, 1) + n_infinite * (0,): H_0_BB,
            **{(0, 0) + tuple(key): value for key, value in H_p_AA.items()},
            **{(0, 1) + tuple(key): value for key, value in H_p_AB.items()},
            **{
                (1, 0) + tuple(key): value.conjugate().transpose()
                for key, value in H_p_AB.items()
            },
            **{(1, 1) + tuple(key): value for key, value in H_p_BB.items()},
        },
        shape=(2, 2),
        n_infinite=n_infinite,
    )

    div_energs = create_div_energs(
        h[zero_index], vecs_a, eigs_a, vecs_b, eigs_b, kpm_params, precalculate_moments
    )

    def unpacked(original: BlockSeries) -> BlockSeries:
        """Unpack the blocks of the series that are stored as explicit matrices"""
        def unpacked_eval(*index):
            res = original.evaluated[index]
            if not all(index[:2]) and not isinstance(res, Zero):
                return res.to_array()
            return res
        return BlockSeries(
            eval=unpacked_eval, shape=(2, 2), n_infinite=original.n_infinite
        )

    return tuple(unpacked(series) for series in general(H, solve_sylvester=div_energs))
