import scipy
import numpy as np
import kwant
from warnings import warn
from functools import partial
from kwant._common import ensure_rng

def build_perturbation(eigenvalues, psi, H0, H1L, H1R=None, params=None, kpm_params=dict()):
    """Build the perturbation elements `<psi_i|H|psi_j>`.

    Given a perturbed Hamiltonian `H`, we calculate the the
    perturbation approximation of the effect of the complement
    space `B` on the space `A`.
    The vectors `psi` expand a space `A`, which complement is `B`.
    `psi` are eigenvectors of `H_0` (not specified) with a
    corresponding set of `eigenvalues`.

    Parameters
    ----------
    eigenvalues : (M) array of floats
        Eigenvalues of the eigenvectors `psi` of `H_0`.
    psi : (M, N) ndarray
        Vectors of length (N), the same as the system defined
        by `ham`.
    H0, H1L, H1R : ndarrays
        Hamiltonian matrix, and perturbations. If H1R=None,
        H1R=H1L is used.
    params : dict, optional
        Parameters for the kwant system.

    Returns
    -------
    ham_ij : (M, M) ndarray
        Matrix elements of the second order perturbation
        of subspace `A` due to the interaction `H` with
        the subspace `B`.
    """
    # Normalize the format of vectors
    eigenvalues = np.atleast_1d(eigenvalues)
    (num_e,) = eigenvalues.shape
    psi = np.atleast_2d(psi)
    num_vecs, dim = psi.shape
    assert num_vecs == num_e
    if H1R is None:
        H1R = H1L
        ReqL = True
    else:
        ReqL = False
    # Normalize the format of the Hamiltonian
    try:
        H0 = scipy.sparse.csr_matrix(H0, dtype=complex)
        H1L = scipy.sparse.csr_matrix(H1L, dtype=complex)
        H1R = scipy.sparse.csr_matrix(H1R, dtype=complex)
    except Exception:
        raise ValueError("'H0' or 'H1L' or 'H1R' is not a matrix.")
    assert H0.shape == H1L.shape
    assert H0.shape == H1R.shape
    assert H0.shape == (dim, dim)

    p_vectors_L = proj((H1L @ psi.T).T, psi)
    p_vectors_R = proj((H1R @ psi.T).T, psi)

    greens = partial(build_greens_function,
                     H0, params=params,
                     kpm_params=kpm_params)

    # evaluate for all the energies
    psi_iR = np.array([greens(vectors=vec)(e) for (vec, e)
                       in zip(p_vectors_R, eigenvalues)]).squeeze(1)
    ham_ij_LR = p_vectors_L.conj() @ psi_iR.T

    if ReqL:
        ham_ij = (ham_ij_LR + ham_ij_LR.conj().T) / 2

    else:
        psi_iL = np.array([greens(vectors=vec)(e) for (vec, e)
                       in zip(p_vectors_L, eigenvalues)]).squeeze(1)
        ham_ij_RL = p_vectors_R.conj() @ psi_iL.T
        ham_ij = (ham_ij_LR + ham_ij_RL.conj().T) / 2

    return ham_ij


def build_greens_function(ham, params=None, vectors=None, kpm_params=dict()):
    """Build a Green's function operator.

    Returns a function that takes a Fermi energy, and returns the
    Green's function of the `vectors` over the occupied energies of the
    Hamiltonian.

    Parameters
    ----------
    ham : kwant.System or ndarray
        Finalized kwant system or ndarray of the Hamiltonian.
    params : dict, optional
        Parameters for the kwant system.
    vectors : ndarray (M, N), optional
        Vectors upon which the projector will act. `M` is the
        number of vectors and `N` the number of orbitals in the
        system.
    kpm_params : dict, optional
        Dictionary containing the parameters to pass to the `~kwant.kpm`
        module. 'num_vectors' will be overwritten to match the number
        of vectors, and 'operator' key will be deleted.
    """
    if vectors is None:
        kpm_params['vector_factory'] = None
    else:
        vectors = np.atleast_2d(vectors)
        num_vectors, dim = vectors.shape
        kpm_params['num_vectors'] = num_vectors
        kpm_params['vector_factory'] = _make_vector_factory(vectors)

    # extract the number of moments or set default to 100
    num_moments = kpm_params.get('num_moments', 100)
    kpm_params['num_moments'] = num_moments
    # prefactors of the kernel in kpm
    m = np.arange(num_moments)
    gs = _kernel(np.ones(num_moments), kernel='J')
    gs[0] = gs[0] / 2

    # overwrite operator to extract kpm expanded vectors only
    kpm_params['operator'] = lambda bra, ket: ket

    # calculate kpm expanded vectors
    spectrum = kwant.kpm.SpectralDensity(ham, params=params, **kpm_params)
    _a, _b = spectrum._a, spectrum._b
    expanded_vectors = np.array(spectrum._moments_list)
    expanded_vectors = np.rollaxis(expanded_vectors, 2, 0)

    def green_expansion(e_F):
        """Takes an energy and returns the Greens function times the vectors,
        for those energies.

        The ndarray returned has initial dimension the same as
        `e_F`, unless it is `1` or `e_F` is a scalar.
        """
        nonlocal _a, _b, m, gs, expanded_vectors, num_moments
        e_F = np.atleast_1d(e_F).flatten()
        (num_e,) = e_F.shape
        e_rescaled = (e_F - _b) / _a
        phi_e = np.arccos(e_rescaled)
        prefactor = -2j / (np.sqrt(1 - e_rescaled**2))
        prefactor = prefactor / _a # rescale energy expansion

        coef = gs * np.exp(-1j * np.outer(phi_e, m))
        coef = prefactor[:,None] * coef
        expanded_vectors_in_energy = (expanded_vectors @ coef.T).T
        if num_e == 1:
            return expanded_vectors_in_energy.squeeze(0)
        return (expanded_vectors @ coef.T).T

    return green_expansion


def exact_greens_function(ham):
    """Takes a Hamiltonian and returns the Green's function operator."""
    eigs, evecs = np.linalg.eigh(ham)
    (dim,) = eigs.shape
    def green(vec, e, eta=1e-2j):
        """Takes a vector `vec` of shape (M,N), with `M` vectors of length `N`,
        the same as the Hamiltonian. Returns the Green's function exact expansion
        of the vectors with the same shape as `vec`."""
        nonlocal dim, eigs, evecs

        # normalize the shapes of `e` and `vec`
        e = np.atleast_1d(e).flatten()
        (num_e,) = e.shape
        vec = np.atleast_2d(vec)
        num_vectors, vec_dim = vec.shape
        assert vec_dim == dim
        assert num_vectors == num_e

        coefs = vec @ evecs.conj()
        e_diff = e[:,None] - eigs[None,:]
        coefs = coefs / (e_diff + eta)
        return coefs @ evecs.T
    return green


def proj(vec, subspace):
    """takes a set of vectors `vec` as an (M,N) ndarray,
    and a subspace (P,N) ndarray, and returns the vectors
    with the subspace projected out.
    The shape of `vec` will be set to a least (1,N), so
    if `M` is omited, a new dimension will be added.
    The output has the same shape as the (reshaped) input.
    """
    vec = np.atleast_2d(vec)
    subspace = np.atleast_2d(subspace)
    assert vec.shape[1] == subspace.shape[1]
    c = subspace.conj() @ vec.T
    return vec - (c.T @ subspace)


def _kernel(moments, kernel='J'):
    """Convolutes the moments with a kernel.

    Implemented for Jackson ('J') and Lorentz ('L') kernels.
    """
    n_moments, *extra_shape = moments.shape
    if kernel == 'J':
        # Jackson kernel, as in Eq. (71), and kernel improved moments,
        # as in Eq. (81).
        m = np.arange(n_moments)
        kernel_array = ((n_moments - m + 1) *
                    np.cos(np.pi * m/(n_moments + 1)) +
                    np.sin(np.pi * m/(n_moments + 1)) /
                    np.tan(np.pi/(n_moments + 1)))
        kernel_array /= n_moments + 1
    elif kernel == 'L':
        # Lorentz kernel
        # parameter for the Lorentz kernel. Prefered values are between 3 and 5
        l = 5
        m = np.arange(n_moments)
        kernel_array = np.sinh(l * (1 - m / n_moments)) / np.sinh(l)

    # transposes handle the case where operators have vector outputs
    conv_moments = np.transpose(moments.transpose() * kernel_array)
    return conv_moments


def _make_vector_factory(vectors=None, eigenvecs=None, rng=None, idx=None):
    """Return a `vector_factory` that outputs vectors.

    Parameters
    ----------
    vectors : iterable of vectors
        Vectors to be returned one by one on call.
    eigenvecs : (M, N) ndarray, optional
        Vectors that expand the space that will be projected out
        of the vectors returned by the `vector_factory`.
    rng : int or array_like, optional
        Seed for the random number generator, or a random number
        generator.
    idx : int, optional
        Set to '-1' or '0' for a modified version of kwant.
        For stable versions, leave the default value set to 'None'.
    """
    if float(kwant.__version__[:3]) > 1.3:
        idx = 0
    else:
        idx = -1
    rng = ensure_rng(rng)
    def vector_factory(n):
        nonlocal idx, rng, vectors, eigenvecs
        if idx == -1:
            idx += 1
            return np.exp(rng.rand(n) * 2j * np.pi)
        vec = vectors[idx]
        if eigenvecs is not None:
            # TODO check this bit
            vec = vec - ((eigenvecs.conj().T @ vec) @ eigenvecs.T).T
        idx += 1
        return vec
    return vector_factory
