import numpy as np
import sympy
from codes.higher_order_lowdin import get_effective_model
from codes.perturbative_model import PerturbativeModel
from ..qsymm.linalg import allclose

def test_simple_model():
    order = 8
    rtol = 2e-2
    # Simple model
    ev2 = np.array([0., 1.])
    mat02 = np.diag(ev2)
    mat12 = {sympy.Symbol('x'): np.array([[0, 1.], [1., 2]])}
    evec2 = np.eye(2)
    indices2 = [0]

    # We know the exact result from series expansion of the exact eigenvalue
    x = sympy.symbols('x')
    exact_result = PerturbativeModel({x**n: np.array([[val]])
                    for n, val in enumerate([0, 0, -1, 2, -3, 2, 6, -28, 61])})

    # Test explicit method
    evec_A = evec2[:, indices2]
    evec_B = evec2[:, np.array([1])]

    model = get_effective_model(mat02, mat12, evec_A, evec_B=evec_B, order=order)

    # Should match within default tolerance
    assert model == exact_result, model - exact_result

    ###
    # Slightly more complicated model
    # Simple model
    ev2 = np.array([0, 0, 1, 1])
    mat02 = np.diag(ev2)
    mat12 = {sympy.Symbol('x'): np.array([[0, 0, 0, 1],
                                          [0, 0, 1, 0],
                                          [0, 1, 2, 0],
                                          [1, 0, 0, 2]])}
    evec2 = np.eye(4)
    indices2 = [0, 1]

    exact_result = PerturbativeModel({x**n: val * np.eye(2)
                                  for n, val in enumerate([0, 0, -1, 2, -3, 2, 6, -28, 61])})

    # Test explicit method
    evec_A = evec2[:, indices2]
    evec_B = evec2[:, np.array([2, 3])]

    model = get_effective_model(mat02, mat12, evec_A, evec_B=evec_B, order=order)

    # Should match within default tolerance
    assert model == exact_result, model - exact_result

    # Test KPM method
    for kpm_params, precalc in [(dict(num_moments=1000), False),
                                (dict(num_moments=1000), True),
                                (dict(energy_resolution=0.01), False)]:
        evec_A = evec2[:, indices2]
        evec_B = None

        model = get_effective_model(mat02, mat12, evec_A, evec_B=evec_B,
                                    order=order, kpm_params=kpm_params,
                                    _precalculate_moments=precalc)

        # Should match wit larger tolerance
        for key in model.keys():
            assert allclose(model[key], exact_result[key], rtol=rtol), (key, model[key] - exact_result[key])
