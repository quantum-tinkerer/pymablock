import scipy
import numpy as np
import kwant
from warnings import warn
from functools import partial
from kwant._common import ensure_rng

def build_perturbation(eigenvalues, psi, ham, params=None):
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
    ham : kwant.System or ndarray
        Finalized kwant system or ndarray of the Hamiltonian.
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
    psi = np.atleast_2d(psi)
    # Normalize the format of 'ham'
    if isinstance(ham, kwant.system.System):
        ham = ham.hamiltonian_submatrix(params=params, sparse=True)
    try:
        ham = scipy.sparse.csr_matrix(ham)
    except Exception:
        raise ValueError("'ham' is neither a matrix nor a Kwant system.")

    def proj_A(vec):
        return (psi.conj() @ vec.T).T @ psi
    def proj_B(vec):
        return vec - (psi.conj() @ vec.T).T @ psi

    # project the vectors to the subspace `B`
    p_vectors = proj_B((ham @ psi.T).T)
    # Build the greens functions for these vectors
    green = build_greens_function(ham, params=params, vectors=p_vectors)
    psi_i = green(eigenvalues)
    # we want only the diagonal elements that map each vector to its eigenvalue
    psi_i = np.diagonal(psi_i, axis1=1, axis2=2)
    """
    This is may be too much of computation, because this evaluates
    the Green's function of each vector to every other energy.
    It could be less expensive to build each green function and evaluate
    it for each energy. Like this
    """
    #greens = lambda vec, e: build_greens_function(
        #ham, params=params, vectors=proj_B(vec)).squeeze()
    #psi_i = np.array([greens(vec, e)
                      #for (vec, e) in zip(vectors, eigenvalues)]).T
    # evaluate for all the energies

    ham_ij = psi_i.conj().T @ psi_i
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
    vectors : ndarray (M,N), optional
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
        num_vectors = vectors.shape[0]
        kpm_params['num_vectors'] = num_vectors
        kpm_params['vector_factory'] = _make_vector_factory(vectors)

    # extract the number of moments or set default to 100
    num_moments = kpm_params.get('num_moments', 100)
    # prefactors of the kernel in kpm
    m = np.arange(num_moments)
    gs = _kernel(np.ones(num_moments))
    gs[0] = gs[0] / 2

    kpm_params['num_moments'] = num_moments
    # overwrite operator to extract kpm expanded vectors only
    kpm_params['operator'] = lambda bra, ket: ket

    # calculate kpm expanded vectors
    spectrum = kwant.kpm.SpectralDensity(ham, params=params, **kpm_params)
    expanded_vectors = np.array(spectrum._moments_list)


    def get_coefs(e_F):
        # TODO put the proper coefficients here
        e_F = np.atleast_1d(e_F)
        e_rescaled = (e_F - spectrum._b) / spectrum._a
        phi_e = np.arccos(e_rescaled)
        prefactor = 2j / np.sqrt(1 - e_rescaled)
        coef = gs * np.exp(-1j * np.outer(phi_e, m))
        coef = prefactor[:, None] * coef
        return coef

    return lambda e_F : get_coefs(e_F) @ expanded_vectors.T


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


def _make_vector_factory(vectors=None, eigenvecs=None, rng=0, idx=-1):
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
    idx : int, default to '-1'
        `idx` is -1 for kwant stable version, if kwant=dev,
        then `idx` is 0.
    """
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
