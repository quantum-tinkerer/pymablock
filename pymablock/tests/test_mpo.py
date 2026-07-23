from dataclasses import dataclass

import numpy as np
import pytest
from scipy.linalg import solve_sylvester
from sympy.physics.quantum import Dagger

from pymablock import block_diagonalize
from pymablock.mpo import (
    BackendMPO,
    SylvesterResult,
    make_mpo_sylvester_solver,
)


@dataclass
class DenseBackend:
    calls: list[str]

    def add(self, left, right):
        self.calls.append("add")
        return left + right

    def scale(self, operator, factor):
        self.calls.append("scale")
        return factor * operator

    def matmul(self, left, right):
        self.calls.append("matmul")
        return left @ right

    def adjoint(self, operator):
        self.calls.append("adjoint")
        return operator.conj().T


def test_backend_mpo_arithmetic():
    backend = DenseBackend([])
    left_array = np.array([[1, 2j], [-2j, 3]])
    right_array = np.array([[2, 1], [1, -1]])
    left = BackendMPO(left_array, backend)
    right = BackendMPO(right_array, backend)

    np.testing.assert_allclose((left + right).operator, left_array + right_array)
    np.testing.assert_allclose((left - right).operator, left_array - right_array)
    np.testing.assert_allclose((-left).operator, -left_array)
    np.testing.assert_allclose((2 * left / 4).operator, left_array / 2)
    np.testing.assert_allclose((left @ right).operator, left_array @ right_array)
    np.testing.assert_allclose(Dagger(left).operator, left_array.conj().T)
    assert backend.calls == [
        "add",
        "scale",
        "add",
        "scale",
        "scale",
        "scale",
        "matmul",
        "adjoint",
    ]


def test_backend_mpo_rejects_incompatible_operations():
    left = BackendMPO(np.eye(2), DenseBackend([]))
    right = BackendMPO(np.eye(2), DenseBackend([]))

    with pytest.raises(ValueError, match="different backends"):
        left + right
    with pytest.raises(ValueError, match="different backends"):
        left @ right
    with pytest.raises(ZeroDivisionError):
        left / 0


def test_sylvester_adapter_selects_blocks_and_checks_residual():
    backend = DenseBackend([])
    blocks = [
        BackendMPO(np.diag([-2.0, -1.0]), backend),
        BackendMPO(np.diag([1.0, 2.0]), backend),
    ]
    rhs = BackendMPO(np.ones((2, 2)), backend)
    seen = []

    def solve(left, right, source, index):
        seen.append((left, right, source, index))
        solution = solve_sylvester(left, -right, source)
        residual = np.linalg.norm(left @ solution - solution @ right - source)
        return SylvesterResult(solution, residual / np.linalg.norm(source), True)

    adapted = make_mpo_sylvester_solver(blocks, solve)
    result = adapted(rhs, (0, 1, 3))

    assert seen[0][0] is blocks[0].operator
    assert seen[0][1] is blocks[1].operator
    assert seen[0][2] is rhs.operator
    assert seen[0][3] == (0, 1, 3)
    np.testing.assert_allclose(
        blocks[0].operator @ result.operator - result.operator @ blocks[1].operator,
        rhs.operator,
    )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (SylvesterResult(np.eye(2), 1e-12, False), "did not converge"),
        (SylvesterResult(np.eye(2), np.nan, True), "non-finite"),
        (SylvesterResult(np.eye(2), 1e-2, True), "exceeds"),
    ],
)
def test_sylvester_adapter_rejects_bad_results(result, message):
    backend = DenseBackend([])
    blocks = [BackendMPO(np.eye(2) * i, backend) for i in (1, 2)]

    def return_result(*_args):
        return result

    adapted = make_mpo_sylvester_solver(
        blocks,
        return_result,
    )

    with pytest.raises(RuntimeError, match=message):
        adapted(BackendMPO(np.eye(2), backend), (0, 1, 1))


def test_backend_mpo_runs_block_diagonalization():
    backend = DenseBackend([])
    left = np.diag([-2.0, -1.0])
    right = np.diag([1.0, 2.0])
    coupling = np.array([[0.2, 0.3j], [-0.1j, 0.4]])
    blocks = [BackendMPO(operator, backend) for operator in (left, right)]

    def solve(left_h_0, right_h_0, rhs, _index):
        solution = solve_sylvester(left_h_0, -right_h_0, rhs)
        residual = left_h_0 @ solution - solution @ right_h_0 - rhs
        return SylvesterResult(
            solution,
            np.linalg.norm(residual) / np.linalg.norm(rhs),
            True,
        )

    H_tilde, U, _ = block_diagonalize(
        [
            [[blocks[0], 0], [0, blocks[1]]],
            [
                [0, BackendMPO(coupling, backend)],
                [BackendMPO(coupling.conj().T, backend), 0],
            ],
        ],
        solve_sylvester=make_mpo_sylvester_solver(blocks, solve),
    )

    assert isinstance(U[0, 1, 1], BackendMPO)
    assert isinstance(H_tilde[0, 0, 2], BackendMPO)

    dense_H_tilde, dense_U, _ = block_diagonalize(
        [
            [[left, 0], [0, right]],
            [[0, coupling], [coupling.conj().T, 0]],
        ]
    )
    np.testing.assert_allclose(U[0, 1, 1].operator, dense_U[0, 1, 1])
    np.testing.assert_allclose(
        H_tilde[0, 0, 2].operator,
        dense_H_tilde[0, 0, 2],
    )
