from itertools import product
from functools import reduce
from operator import add

import numpy as np
from scipy.sparse.linalg import aslinearoperator

import pdb

from lowdin.linalg import ComplementProjector, complement_projected
from lowdin.kpm_funcs import greens_function
from lowdin.block_diagonalization import general
from lowdin.series import BlockSeries, zero

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
        #pdb.set_trace()
        Y = Y @ np.eye(Y.shape[-1])
        if need_kpm and need_explicit:
            result = sylvester_kpm(Y) + sylvester_explicit(Y)
        elif need_kpm:
            result = sylvester_kpm(Y)
        elif need_explicit:
            result = sylvester_explicit(Y)

        return aslinearoperator(result)

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

    h_0_aa = aslinearoperator(np.diag(eigs_a))
    h_0_bb = complement_projected(h[zero_index], vecs_a)

    h_p_aa = {k: aslinearoperator(vecs_a.conj().T @ v @ vecs_a)for k, v in h.items()if any(k)}
    h_p_bb = {k: complement_projected(v, vecs_a)for k, v in h.items() if any(k)}
    h_p_ab = {k: aslinearoperator(vecs_a.conj().T @ v @ p_b) for k, v in h.items() if any(k)}

    H = BlockSeries(
        data={
            (0, 0) + n_infinite * (0,): h_0_aa,
            (1, 1) + n_infinite * (0,): h_0_bb,
            **{(0, 0) + tuple(key): value for key, value in h_p_aa.items()},
            **{(0, 1) + tuple(key): value for key, value in h_p_ab.items()},
            **{
                (1, 0) + tuple(key): value.adjoint()
                for key, value in h_p_ab.items()
            },
            **{(1, 1) + tuple(key): value for key, value in h_p_bb.items()},
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
            if not all(index[:2]) and zero != res:
                return res @ np.eye(res.shape[-1])
            return res
        return BlockSeries(
            eval=unpacked_eval, shape=(2, 2), n_infinite=original.n_infinite
        )

    return tuple(unpacked(series) for series in general(H, solve_sylvester=div_energs))
