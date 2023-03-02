# # The polynomial alternative to Lowdin perturbation theory
#
# See [this hackmd](https://hackmd.io/Rpt2C8oOQ2SGkGS9OYrlfQ?view) for the motivation and the expressions

# +
from itertools import product
from functools import reduce
from operator import matmul

import numpy as np
from sympy.physics.quantum import Dagger
import tinyarray as ta

from lowdin.series import BlockOperatorSeries, Zero, _zero_sum, _zero, cauchy_dot_product


# -


def generate_volume(wanted_orders):
    """
    Generate ordered array with tinyarrays in volume of wanted_orders.

    wanted_orders : list of tinyarrays containing the wanted order of each perturbation.

    Returns:
    List of sorted tinyarrays contained in the volume of required orders to compute the wanted orders.
    """
    wanted_orders = np.array(wanted_orders)
    N_o, N_p = wanted_orders.shape
    max_order = np.max(wanted_orders, axis=0)
    possible_orders = np.array(
        np.meshgrid(*(np.arange(order + 1) for order in max_order))
    ).reshape(len(max_order), -1)
    indices = np.any(
        np.all(
            possible_orders.reshape(N_p, -1, 1) <= wanted_orders.T.reshape(N_p, 1, -1),
            axis=0,
        ),
        axis=1,
    )
    keep_arrays = possible_orders.T[indices]
    return (ta.array(i) for i in sorted(keep_arrays, key=sum) if any(i))


def product_by_order(order, *terms, op=None, hermitian=False):
    """
    Compute sum of all product of terms of wanted order.

    order : int or tinyarray containing the order of the product.
    terms : list of dictionaries of terms to multiply.
    op : (optional) callable for multiplying terms.
    hermitian : (optional) bool for whether to compute hermitian conjugate.

    Returns:
    Sum of all products of terms of wanted order.
    """
    if op is None:
        op = matmul
    contributing_products = []
    for combination in product(*(term.items() for term in terms)):
        key = tuple(key for key, _ in combination)
        if sum(key) != order:
            continue
        values = [value for _, value in combination if value is not None]
        if any(isinstance(value, Zero) for value in values):
            continue
        if hermitian and key > tuple(reversed(key)):
            # exclude half of the reversed partners to prevent double counting
            continue
        temp = reduce(op, values)
        if hermitian and key == tuple(reversed(key)):
            temp /= 2
        contributing_products.append(temp)
    result = _zero_sum(contributing_products)
    if hermitian and not isinstance(result, Zero):
        result += Dagger(result)
    return result

def compute_next_orders(
    H_0_AA,
    H_0_BB,
    H_p_AA,
    H_p_BB,
    H_p_AB,
    wanted_orders,
    divide_energies=None,
    *,
    op=None
):
    """
    Computes transformation to diagonalized Hamiltonian with multivariate perturbation.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    wanted_orders : list of tinyarrays containing the wanted order of each perturbation
    divide_energies : (optional) callable for solving Sylvester equation
    op: callable for multiplying terms

    Returns:
    exp_S : np.array of the transformation to diagonalized Hamiltonian
    """

    zero_index = ta.zeros([len(wanted_orders[0])], int)
    if any(zero_index in pert for pert in (H_p_AA, H_p_AB, H_p_BB)):
        raise ValueError("Perturbation terms may not contain zeroth order")

    H_p_BA = {key: Dagger(value) for key, value in H_p_AB.items()}
    H = np.array(
        [
            [{zero_index: H_0_AA, **H_p_AA}, H_p_AB],
            [H_p_BA, {zero_index: H_0_BB, **H_p_BB}],
        ],
        dtype=object,
    )

    # We use None as a placeholder for identity.
    exp_S = np.array([[{zero_index: None}, {}], [{}, {zero_index: None}]], dtype=object)

    if divide_energies is None:
        # The Hamiltonians must already be diagonalized
        E_A = np.diag(H_0_AA)
        E_B = np.diag(H_0_BB)
        if not np.allclose(H_0_AA, np.diag(E_A)):
            raise ValueError("H_0_AA must be diagonal")
        if not np.allclose(H_0_BB, np.diag(E_B)):
            raise ValueError("H_0_BB must be diagonal")
        energy_denominators = 1 / (E_A.reshape(-1, 1) - E_B)

        def divide_energies(Y):
            return Y * energy_denominators

    needed_orders = generate_volume(wanted_orders)

    for order in needed_orders:
        Y = _zero_sum(
            -((-1) ** i)
            * product_by_order(order, exp_S[0, i], H[i, j], exp_S[j, 1], op=op)
            for i in (0, 1)
            for j in (0, 1)
        )
        if not isinstance(Y, Zero):
            exp_S[0, 1][order] = divide_energies(Y)
            exp_S[1, 0][order] = -Dagger(exp_S[0, 1][order])

        for i in (0, 1):
            exp_S[i, i][order] = (
                _zero_sum(
                    (
                        -product_by_order(
                            order, exp_S[i, i], exp_S[i, i], op=op, hermitian=True
                        ),
                        product_by_order(
                            order,
                            exp_S[i, 1 - i],
                            exp_S[1 - i, i],
                            op=op,
                            hermitian=True,
                        ),
                    )
                )
                / 2
            )

    return exp_S


def H_tilde(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB, exp_S, op=None):
    """
    Computes perturbed Hamiltonian in eigenbasis of unperturbed Hamiltonian.

    H_0_AA : np.array of the unperturbed Hamiltonian of subspace AA
    H_0_BB : np.array of the unperturbed Hamiltonian of subspace BB
    H_p_AA : dictionary of perturbation terms of subspace AA
    H_p_BB : dictionary of perturbation terms of subspace BB
    H_p_AB : dictionary of perturbation terms of subspace AB
    exp_S : np.array of the transformation to diagonalized Hamiltonian

    Returns:
    H_tilde : BlockOperatorSeries
    """

    if op is None:
        op = matmul

    n_infinite = len(list(exp_S[0, 0].keys())[0])
    zeroth_order = (0,) * n_infinite
    H = BlockOperatorSeries.from_dict(
        {
            **{
                (0, 0) + zeroth_order: H_0_AA,
            },
            **{
                (1, 1) + zeroth_order: H_0_BB,
            },
            **{
                (0, 0) + tuple(key): value
                for key, value in H_p_AA.items()
            },
            **{
                (0, 1) + tuple(key): value
                for key, value in H_p_AB.items()
            },
            **{
                (1, 0) + tuple(key): Dagger(value)
                for key, value in H_p_AB.items()
            },
            **{
                (1, 1) + tuple(key): value
                for key, value in H_p_BB.items()
            },
        },
        shape=(2, 2), n_infinite=n_infinite
    )

    exp_S = BlockOperatorSeries.from_dict(
        {
            **{
                (0, 0) + tuple(key): value
                for key, value in exp_S[0, 0].items()
            },
            **{
                (0, 1) + tuple(key): value 
                for key, value in exp_S[0, 1].items()
            },
            **{
                (1, 0) + tuple(key): -Dagger(value) 
                for key, value in exp_S[0, 1].items()
            },
            **{
                (1, 1) + tuple(key): value
                for key, value in exp_S[1, 1].items()
            },
        },
        shape=(2, 2), n_infinite=n_infinite
    )
    exp_S_dagger = BlockOperatorSeries(
        eval=(
            lambda entry: exp_S.evaluated[entry] if entry[0] == entry[1] else -exp_S.evaluated[entry]
        ),
        shape=(2, 2),
        n_infinite=n_infinite,
    )

    return cauchy_dot_product(exp_S_dagger, H, exp_S, op=op, hermitian=True)