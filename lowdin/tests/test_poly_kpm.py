from itertools import product, count

import pytest
import numpy as np
import tinyarray as ta
from scipy.linalg import eigh, block_diag

from lowdin.poly_kpm import SumOfOperatorProducts, create_div_energs, numerical
from lowdin.linalg import ComplementProjector


@pytest.fixture(
    scope="module",
    params=[
        [[3]],
        [[2, 2]],
        [[3, 1], [1, 3]],
        [[2, 2, 2], [3, 0, 0]],
    ],
)
def wanted_orders(request):
    """
    Return a list of orders to compute.
    """
    return request.param


@pytest.fixture(scope="module")
def Ns():
    """
    Return a random number of states for each block (A, B).
    """
    return np.random.randint(1, high=5, size=2)


@pytest.fixture(scope="module")
def hamiltonians(Ns, wanted_orders):
    """
    Produce random Hamiltonians to test.

    Ns: dimension of each block (A, B)
    wanted_orders: list of orders to compute

    Returns:
    hams: list of Hamiltonians
    """
    N_p = len(wanted_orders[0])
    orders = ta.array(np.eye(N_p))
    hams = []
    for i in range(2):
        hams.append(np.diag(np.sort(np.random.rand(Ns[i])) - i))

    def matrices_it(N_i, N_j, hermitian):
        """
        Generate random matrices of size N_i x N_j.

        N_i: number of rows
        N_j: number of columns
        hermitian: if True, the matrix is hermitian

        Returns:
        generator of random matrices
        """
        for i in count():
            H = np.random.rand(N_i, N_j) + 1j * np.random.rand(N_i, N_j)
            if hermitian:
                H += H.conj().T
            yield H

    for i, j, hermitian in zip([0, 1, 0], [0, 1, 1], [True, True, False]):
        matrices = matrices_it(Ns[i], Ns[j], hermitian)
        hams.append({order: matrix for order, matrix in zip(orders, matrices)})
    return hams


def random_term(n, m, length, start, end, rng=None):
    """Generate a random term.

    Parameters
    ----------
    n : int
        Size of "A" space
    m : int
        Size of "B" space
    length : int
        Number of operators in the term
    start, end : str
        Start and end spaces of the term (A or B)
    rng : np.random.Generator
        Random number generator
    """
    if rng is None:
        rng = np.random.default_rng()
    spaces = "".join(np.random.choice(a=["A", "B"], size=length - 1))
    spaces = start + spaces + end
    op_spaces = ["".join(s) for s in zip(spaces[:-1], spaces[1:])]
    op_dims = [
        (n if dim[0] == "A" else m, m if dim[1] == "B" else n) for dim in op_spaces
    ]
    ops = [rng.random(size=dim) for dim in op_dims]
    return SumOfOperatorProducts([[(op, space) for op, space in zip(ops, op_spaces)]])


def assert_almost_zero(a, decimal=5, extra_msg=""):
    """
    Assert that all values in a are almost zero.

    a: array to check
    decimal: number of decimal places to check
    extra_msg: extra message to print if assertion fails
    """
    for key, value in a.items():
        np.testing.assert_almost_equal(
            value, 0, decimal=decimal, err_msg=f"{key=} {extra_msg}"
        )


# ############################################################################################


def test_shape_validation():
    """Test that only terms of compatible shapes are accepted.

    Instead of providing terms manually we rely on SumOfOperatorProducts
    creating new instances of itself on addition and multiplication.
    """
    n, m = 4, 10
    terms = {
        "AA": random_term(n, m, 1, "A", "A"),
        "AB": random_term(n, m, 1, "A", "B"),
        "BA": random_term(n, m, 1, "B", "A"),
        "BB": random_term(n, m, 1, "B", "B"),
    }
    for (space1, term1), (space2, term2) in product(terms.items(), repeat=2):
        # Sums should work if the spaces are the same
        if space1 == space2:
            # no error, moreover the result should simplify to a single term
            term1 + term2
            assert len(term1.terms) == 1
        else:
            with pytest.raises(ValueError):
                term1 + term2

        # Matmuls should work if start space of term2 matches end space of term1
        if space1[1] == space2[0]:
            term1 @ term2
        else:
            with pytest.raises(ValueError):
                term1 @ term2


def test_neg():
    """Test that negation works."""
    n, m = 4, 10
    term = random_term(n, m, 1, "A", "A")
    zero = term + -term
    # Should have one term with all zeros
    assert len(zero.terms) == 1
    np.testing.assert_allclose(zero.terms[0][0][0], 0)


def test_create_div_energs_kpm(hamiltonians):
    n_a = hamiltonians[0].shape[0]
    n_b = hamiltonians[1].shape[0]
    h_0 = block_diag(hamiltonians[0], hamiltonians[1])
    eigs, vecs = eigh(h_0)
    eigs_a, vecs_a = eigs[:n_a], vecs[:, :n_a]
    eigs_b, vecs_b = eigs[n_a:], vecs[:, n_a:]

    Y = []
    for _ in range(5):
        h_ab = np.random.random((n_a + n_b, n_a + n_b)) + 1j * np.random.random(
            (n_a + n_b, n_a + n_b)
        )
        h_ab += h_ab.conj().T
        Y.append(vecs_a.conj().T @ h_ab @ ComplementProjector(vecs_a))

    de_kpm_func = lambda Y: create_div_energs(
        h_0, vecs_a, eigs_a, kpm_params=dict(num_moments=1000)
    )(Y)
    de_exact_func = lambda Y: create_div_energs(h_0, vecs_a, eigs_a, vecs_b, eigs_b)(Y)

    # apply h_ab from left -> Y.conj() since G_0 is hermitian
    applied_exact = [de_exact_func(y.conj()) for y in Y]
    applied_kpm = [de_kpm_func(y.conj()) for y in Y]

    diff_approach = {
        i: np.abs(applied_exact[i] - applied_kpm[i]) for i in range(len(Y))
    }

    assert_almost_zero(diff_approach, decimal=1, extra_msg="")


def test_create_div_energs_kpm(hamiltonians):
    n_a = hamiltonians[0].shape[0]
    n_b = hamiltonians[1].shape[0]
    h_0 = block_diag(hamiltonians[0], hamiltonians[1])
    eigs, vecs = eigh(h_0)
    eigs_a, vecs_a = eigs[:n_a], vecs[:, :n_a]
    eigs_b, vecs_b = eigs[n_a:], vecs[:, n_a:]

    Y = []
    for _ in range(5):
        h_ab = np.random.random((n_a + n_b, n_a + n_b)) + 1j * np.random.random(
            (n_a + n_b, n_a + n_b)
        )
        h_ab += h_ab.conj().T
        """
        Y.append(
            SumOfOperatorProducts(
                [[(vecs_a.conj().T @ h_ab @ ComplementProjector(vecs_a), "AB")]]
            )
        )
        """
        Y.append(
            SumOfOperatorProducts(
                [[((h_ab @ ComplementProjector(vecs_a))[:n_a, :], "AB")]]
            )
        )

    de_kpm_func = lambda Y: create_div_energs(
        h_0, vecs_a, eigs_a, kpm_params=dict(num_moments=1000)
    )(Y)
    de_exact_func = lambda Y: create_div_energs(h_0, vecs_a, eigs_a, vecs_b, eigs_b)(Y)

    # apply h_ab from left -> Y.conj() since G_0 is hermitian
    applied_exact = [de_exact_func(y.conjugate()) for y in Y]
    applied_kpm = [de_kpm_func(y.conjugate()) for y in Y]

    diff_approach = {
        i: np.abs(applied_exact[i].to_array() - applied_kpm[i].to_array())
        for i in range(len(Y))
    }

    assert_almost_zero(diff_approach, decimal=1, extra_msg="")


def test_ab_is_zero():
    max_ord = 4
    n_dim = np.random.randint(low=20, high=100)
    n_a = np.random.randint(low=0, high=int(n_dim / 2))
    h_0 = np.diag(np.sort(np.random.random(n_dim)))
    h_0[n_a:, n_a:] = 15 * h_0[n_a:, n_a:] # increase gap for test to succeed in more cases
    eigs_a = np.diag(h_0)[:n_a]
    vecs_a = np.eye(n_dim)[:, :n_a]

    h_p = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
    h_p += h_p.conjugate().transpose()

    ham = {(0,): h_0, (1,): h_p}

    h_t, u, u_adj = numerical(
        ham, vecs_a, eigs_a, kpm_params={"num_moments": 10000}
    )

    ab_s = h_t.evaluated[0, 1, :max_ord]
    ab_s = list(ab_s[~ab_s.mask].data)
    ab_s = {i: ab_s[i] for i in range(len(ab_s))}

    assert_almost_zero(ab_s, decimal=1, extra_msg="")
