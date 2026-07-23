from dataclasses import dataclass

import numpy as np
import pytest

from pymablock import block_diagonalize
from pymablock.implicit import StateSolveResult, block_diagonalize_implicit


@dataclass
class DenseStateBackend:
    calls: list[str]

    def apply(self, operator, state):
        self.calls.append("apply")
        return operator @ state

    def add_states(self, left, right):
        self.calls.append("add_states")
        return left + right

    def scale_state(self, state, factor):
        self.calls.append("scale_state")
        return factor * state

    def inner(self, left, right):
        self.calls.append("inner")
        return np.vdot(left, right)

    def adjoint_operator(self, operator):
        self.calls.append("adjoint_operator")
        return operator.conj().T


def dense_shifted_solver(h_0, energy, rhs, references, _index, _column):
    vectors = np.column_stack(references)
    projector = np.eye(h_0.shape[0]) - vectors @ vectors.conj().T
    shifted = projector @ (h_0 - energy * np.eye(h_0.shape[0])) @ projector
    constrained = shifted + vectors @ vectors.conj().T
    solution = projector @ np.linalg.solve(constrained, rhs)
    residual = shifted @ solution - rhs
    relative_residual = np.linalg.norm(residual) / np.linalg.norm(rhs)
    return StateSolveResult(solution, relative_residual, True, sweeps=1)


def test_implicit_state_block_diagonalization_matches_dense():
    h_0 = np.diag([-2.0, -1.0, 1.0, 3.0])
    perturbation = np.array(
        [
            [0.1, 0.2, 0.3j, -0.2],
            [0.2, -0.1, 0.4, 0.1j],
            [-0.3j, 0.4, 0.3, -0.2j],
            [-0.2, -0.1j, 0.2j, -0.3],
        ]
    )
    references = tuple(np.eye(4)[:, index] for index in range(2))
    backend = DenseStateBackend([])

    implicit_h_tilde, implicit_u, _ = block_diagonalize_implicit(
        [h_0, perturbation],
        references,
        backend,
        dense_shifted_solver,
        max_relative_residual=1e-12,
    )
    dense_h_tilde, dense_u, _ = block_diagonalize(
        [h_0, perturbation],
        subspace_indices=np.array([0, 0, 1, 1]),
    )

    np.testing.assert_allclose(
        implicit_h_tilde[0, 0, 2].dense,
        dense_h_tilde[0, 0, 2],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.column_stack(implicit_u[1, 0, 1].states)[2:],
        dense_u[1, 0, 1],
        atol=1e-12,
    )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (StateSolveResult(np.ones(3), 1e-12, False), "did not converge"),
        (StateSolveResult(np.ones(3), np.nan, True), "non-finite"),
        (StateSolveResult(np.ones(3), 1e-2, True), "exceeds"),
    ],
)
def test_implicit_state_solver_rejects_bad_results(result, message):
    h_0 = np.diag([-1.0, 1.0, 2.0])
    perturbation = np.ones((3, 3))
    reference = np.array([1.0, 0.0, 0.0])
    backend = DenseStateBackend([])

    def return_result(*_args):
        return result

    h_tilde, _, _ = block_diagonalize_implicit(
        [h_0, perturbation],
        [reference],
        backend,
        return_result,
        max_relative_residual=1e-8,
    )
    with pytest.raises(RuntimeError) as caught:
        _ = h_tilde[0, 0, 2]
    causes = []
    error = caught.value
    while error is not None:
        causes.append(str(error))
        error = error.__cause__
    assert any(message in cause for cause in causes)


def test_implicit_state_rejects_non_eigenstate_references():
    h_0 = np.diag([-1.0, 1.0])
    reference = np.array([1.0, 1.0]) / np.sqrt(2)
    backend = DenseStateBackend([])

    with pytest.raises(ValueError, match="eigenstates"):
        block_diagonalize_implicit(
            [h_0, np.eye(2)],
            [reference],
            backend,
            dense_shifted_solver,
        )
