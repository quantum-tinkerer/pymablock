import pytest
import sympy
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp

from pymablock import block_diagonalize
from pymablock.number_ordered_form import NumberOperator, NumberOrderedForm

# Keep the coupler parameters symbolic so the regression tracks the general
# expression growth of the operator pipeline rather than one concrete point.
E_1, E_2, E_C, G_1C, G_2C, G_12, W_1, W_2, W_C = sympy.symbols(
    "E_1 E_2 E_C G_1C G_2C G_12 W_1 W_2 W_C",
    real=True,
)
MAX_PERTURBATION_ORDER = 4


def _symbolic_tunable_coupler_orders(wc_value):
    a_1, a_2, a_c = BosonOp("a_1"), BosonOp("a_2"), BosonOp("a_c")
    n_1, n_2, n_c = (NumberOperator(op) for op in (a_1, a_2, a_c))

    H_0 = (
        W_1 * n_1
        + W_2 * n_2
        + wc_value * n_c
        + E_1 * n_1 * (n_1 - 1) / 2
        + E_2 * n_2 * (n_2 - 1) / 2
        + E_C * n_c * (n_c - 1) / 2
    )
    H_1 = sympy.S.Zero
    for g, a_l, a_r in (
        (G_1C, a_1, a_c),
        (G_2C, a_2, a_c),
        (G_12, a_1, a_2),
    ):
        H_1 += g * (
            a_l * Dagger(a_r) + Dagger(a_l) * a_r - a_l * a_r - Dagger(a_l) * Dagger(a_r)
        )

    H_tilde, *_ = block_diagonalize([H_0, H_1])
    return [H_tilde[0, 0, order] for order in range(MAX_PERTURBATION_ORDER + 1)]


def _number_ordered_form_metrics(expr) -> dict[str, int]:
    if not isinstance(expr, NumberOrderedForm):
        return {
            "terms": 0 if expr == sympy.S.Zero else 1,
            "coeff_ops": int(sympy.count_ops(expr)),
            "coeff_str_len": len(str(expr)),
        }

    coeffs = [coeff for _, coeff in expr.args[1]]
    return {
        "terms": len(coeffs),
        "coeff_ops": int(sum(sympy.count_ops(coeff) for coeff in coeffs)),
        "coeff_str_len": sum(len(str(coeff)) for coeff in coeffs),
    }


@pytest.mark.no_cover
def test_tunable_coupler_symbolic_output_regression(data_regression):
    orders = _symbolic_tunable_coupler_orders(W_C)
    H_eff = sum(orders, start=sympy.S.Zero)
    assert isinstance(H_eff, NumberOrderedForm)

    data_regression.check(
        {
            "parameterization": "symbolic",
            "max_perturbation_order": MAX_PERTURBATION_ORDER,
            **{
                f"order_{order}": _number_ordered_form_metrics(expr)
                for order, expr in enumerate(orders)
            },
            "effective": _number_ordered_form_metrics(H_eff),
        }
    )
