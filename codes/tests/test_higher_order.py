import numpy as np
import sympy
from codes.higher_order_lowdin import PerturbativeModel, get_effective_model, allclose


def test_simple_model():
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

    model = get_effective_model(mat02, mat12, evec_A, evec_B, order=8)

    # Should match within default tolerance
    assert model == exact_result, model - exact_result

    # Test KPM method
    kpm_params = dict(num_moments=1000)
    evec_A = evec2[:, indices2]
    evec_B = None

    model = get_effective_model(mat02, mat12, evec_A, evec_B, order=8, kpm_params=kpm_params)

    # Should match wit larger tolerance
    for key in model.keys():
        assert allclose(model[key], exact_result[key], rtol=2e-2), (key, model[key] - exact_result[key])
