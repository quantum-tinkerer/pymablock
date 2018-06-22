import scipy
import numpy as np
import kwant
from warnings import warn
from functools import partial
from kwant._common import ensure_rng



def proj(vec, subspace):
    """Project out "subspace" from "vec".

    Parameters
    ----------
    vec : array(N)
        Vector to which project P_B obtained from "subspace" is applied.
    subspace : array(N, M)
        Subspace in numpy convention: subspace[:, i] is i-th vector.
        These vectors are used to built project P_A = sum_i |i X i|,
        from which project P_B = identity(N, N) - P_A is built.

    Returns
    -------
    vec : array(N)
    """
    P_A = subspace @ subspace.T.conj()
    return vec -  P_A @ vec


def build_perturbation(ev, evec, H0, H1L, H1R=None, indices=None,
                       kpm_params=None):
    """Build the perturbation elements of the 2nd order perturbation.

    This calculates "H1L'_{im} H1R'_{mj} x (1 / (Ei - Em) + 1 / (Ej / Em))"".

    Given a perturbed Hamiltonian "H0", we calculate the the
    perturbation approximation of the effect of the complement
    space "B" on the space "A".
    The vectors "evec[:, indices]" expand a space "A", which complement is "B".
    Space "B" consists of subspace "B1" and "B2".
    Subspace B1 contains eigenvectors "evec[:, i]" for "i" not in "indices"
    and is not considered by this function.
    Subspace "B2" contains all eingestates of "H0" not included in "evec" that
    are considered approximately by KPM through this function.

    Parameters
    ----------
    ev : array(M)
        Eigenvalues of "H0" for states known exactly.
    evec : (N, M) ndarray
        Eigenvectors of "H0" for states known exactly.
    H0, H1L, H1R : ndarrays
        Hamiltonian matrix, and perturbations. If H1R=None,
        H1R=H1L is used.
    indices : sequence of M integers
        Indices of states for which we calculate the effective model.
        If unset (None) then all states in "evec" will be considered.

    Returns
    -------
    ham_ij : (M, M) ndarray
        Matrix elements of the second order perturbation
        of subspace `A` due to the interaction `H` with
        the subspace `B`.
    """
    if kpm_params is None:
        kpm_params = dict()

    if indices is None:
        indices = range(len(ev))

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

    # Debug checks (to be removed later or replaced)
    assert len(ev) == evec.shape[1]
    assert len(indices) <= len(ev)
    assert H0.shape == H1L.shape
    assert H0.shape == H1R.shape

    # Project out everything from inside "evec" subspace.
    p_vectors_L = proj(H1L @ evec[:, indices], evec)
    p_vectors_R = proj(H1R @ evec[:, indices], evec)
    ev = ev[indices]

    greens = partial(build_greens_function, H0, kpm_params=kpm_params)

    # evaluate for all the energies
    psi_iR = np.array([greens(vectors=vec)(e) for (vec, e)
                       in zip(p_vectors_R.T, ev)]).squeeze(1)
    ham_ij_LR = p_vectors_L.T.conj() @ psi_iR.T

    if ReqL:
        ham_ij = (ham_ij_LR + ham_ij_LR.conj().T) / 2

    else:
        psi_iL = np.array([greens(vectors=vec)(e) for (vec, e)
                           in zip(p_vectors_L.T, ev)]).squeeze(1)
        ham_ij_RL = p_vectors_R.T.conj() @ psi_iL.T
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


def _make_vector_factory(vectors=None, eigenvecs=None, rng=0):
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
    """
    idx = -1 + 1*_version_higher() # initial vector index according to version

    rng = ensure_rng(rng)
    def vector_factory(n):
        nonlocal idx, rng, vectors, eigenvecs
        if idx == -1:
            idx += 1
            return np.exp(rng.rand(n) * 2j * np.pi)
        vec = vectors[idx]
        if eigenvecs is not None:
            vec = vec - ((eigenvecs.conj() @ vec) @ eigenvecs)
        idx += 1
        return vec
    return vector_factory

def _version_higher(v='1.3.2'):
    from kwant import __version__ as n
    v = tuple(int(char) for char in  v[:5].split('.'))
    n = tuple(int(char) for char in  n[:5].split('.'))
    if n > v:
        return True
    return False
