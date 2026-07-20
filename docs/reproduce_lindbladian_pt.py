"""Reproduce representative results from Lindbladian perturbation theory."""

import numpy as np
from scipy.linalg import eig, eigvals, null_space, orth, solve_sylvester
from scipy.optimize import linear_sum_assignment

from pymablock import block_diagonalize
from pymablock.series import zero


def coherent_superoperator(hamiltonian):
    """Return -i[H, .] in row-major vectorization."""
    identity = np.eye(hamiltonian.shape[0])
    return -1j * (np.kron(hamiltonian, identity) - np.kron(identity, hamiltonian.T))


def dissipator_superoperator(jump):
    """Return D[J] in row-major vectorization."""
    identity = np.eye(jump.shape[0])
    jump_norm = jump.conj().T @ jump
    return np.kron(jump, jump.conj()) - 0.5 * (
        np.kron(jump_norm, identity) + np.kron(identity, jump_norm.T)
    )


def kron(*operators):
    """Kronecker product of a sequence of operators."""
    result = np.array([[1]], dtype=complex)
    for operator in operators:
        result = np.kron(result, operator)
    return result


def structure_residuals(liouvillian, hilbert_dimension):
    """Return trace- and Hermiticity-preservation residuals."""
    trace = np.eye(hilbert_dimension).reshape(1, -1)
    dagger = np.zeros_like(liouvillian)
    for row in range(hilbert_dimension):
        for column in range(hilbert_dimension):
            dagger[column * hilbert_dimension + row, row * hilbert_dimension + column] = 1
    return (
        np.linalg.norm(trace @ liouvillian),
        np.linalg.norm(dagger @ liouvillian.conj() - liouvillian @ dagger),
    )


def reiter_sorensen_lambda_system():
    """Compare Pymablock with the effective-operator formalism at order two."""
    dimension = 3
    detuning = 2.3
    decay_rates = (0.7, 1.1)
    rabi_frequencies = (0.8, 0.55)

    basis = np.eye(dimension, dtype=complex)

    def matrix_unit(row, column):
        return np.outer(basis[:, row], basis[:, column].conj())

    h_0 = detuning * matrix_unit(2, 2)
    perturbation = 0.5 * (
        rabi_frequencies[0] * (matrix_unit(2, 0) + matrix_unit(0, 2))
        + rabi_frequencies[1] * (matrix_unit(2, 1) + matrix_unit(1, 2))
    )
    jumps = [np.sqrt(decay_rates[index]) * matrix_unit(index, 2) for index in range(2)]

    liouvillian_0 = coherent_superoperator(h_0)
    for jump in jumps:
        liouvillian_0 += dissipator_superoperator(jump)
    liouvillian_1 = coherent_superoperator(perturbation)

    eigenvalues, right_eigenvectors = eig(liouvillian_0)
    order = np.argsort(np.abs(eigenvalues))
    eigenvalues = eigenvalues[order]
    right_eigenvectors = right_eigenvectors[:, order]
    inverse_eigenvectors = np.linalg.inv(right_eigenvectors)

    assert np.max(np.abs(eigenvalues[:4])) < 1e-12
    transformed_perturbation = inverse_eigenvectors @ liouvillian_1 @ right_eigenvectors
    effective, *_ = block_diagonalize(
        [np.diag(eigenvalues), transformed_perturbation],
        subspace_indices=np.array([0] * 4 + [1] * 5),
        hermitian=False,
    )
    pymablock_order_2 = np.asarray(effective[0, 0, 2])

    total_decay = sum(decay_rates)
    h_nh = detuning - 0.5j * total_decay
    excitation = 0.5 * (
        rabi_frequencies[0] * matrix_unit(2, 0) + rabi_frequencies[1] * matrix_unit(2, 1)
    )
    deexcitation = excitation.conj().T
    effective_hamiltonian = (
        -0.5 * ((1 / h_nh) + (1 / h_nh).conjugate()) * (deexcitation @ excitation)
    )
    effective_jumps = [jump @ ((1 / h_nh) * excitation) for jump in jumps]
    published_order_2 = coherent_superoperator(effective_hamiltonian)
    for jump in effective_jumps:
        published_order_2 += dissipator_superoperator(jump)

    published_slow_block = (
        inverse_eigenvectors @ published_order_2 @ right_eigenvectors
    )[:4, :4]
    coefficient_error = np.max(np.abs(pymablock_order_2 - published_slow_block))
    assert coefficient_error < 2e-14

    ground_indices = [0, 1, 3, 4]
    published_ground_block = published_order_2[np.ix_(ground_indices, ground_indices)]
    effective_eigenvalues = eigvals(published_ground_block)
    couplings = np.array([0.2, 0.14, 0.1, 0.07, 0.05])
    spectral_errors = []
    for coupling in couplings:
        exact = eigvals(liouvillian_0 + coupling * liouvillian_1)
        exact = exact[np.argsort(np.abs(exact))[:4]]
        approximate = coupling**2 * effective_eigenvalues
        exact_indices, approximate_indices = linear_sum_assignment(
            np.abs(exact[:, None] - approximate[None, :])
        )
        spectral_errors.append(
            np.max(np.abs(exact[exact_indices] - approximate[approximate_indices]))
        )
    error_power = np.polyfit(np.log(couplings), np.log(spectral_errors), 1)[0]
    assert abs(error_power - 4) < 0.02

    print("Reiter-Sorensen Lambda system")
    print(f"  second-order coefficient error: {coefficient_error:.3e}")
    print(f"  low-spectrum remainder scales as g^{error_power:.4f}")


def macieszczak_metastable_three_level_system():
    """Reproduce the detached slow pair in the three-level shelving model."""
    dimension = 3
    omega_1 = 1.0
    kappa = 4 * omega_1
    omega_2 = 0.1 * omega_1
    basis = np.eye(dimension, dtype=complex)

    def matrix_unit(row, column):
        return np.outer(basis[:, row], basis[:, column].conj())

    h_0 = omega_1 * (matrix_unit(1, 0) + matrix_unit(0, 1))
    h_1 = matrix_unit(2, 0) + matrix_unit(0, 2)
    jump = np.sqrt(kappa) * matrix_unit(0, 1)
    liouvillian_0 = coherent_superoperator(h_0) + dissipator_superoperator(jump)
    liouvillian_1 = coherent_superoperator(h_1)

    # At the paper's parameters the fast block is not safely diagonalized.
    # The zero eigenspace and range of L0 nevertheless give complementary
    # invariant slow and fast subspaces.
    slow_basis = null_space(liouvillian_0)
    fast_basis = orth(liouvillian_0)
    basis_change = np.hstack((slow_basis, fast_basis))
    inverse_basis = np.linalg.inv(basis_change)
    transformed_0 = inverse_basis @ liouvillian_0 @ basis_change
    transformed_1 = inverse_basis @ liouvillian_1 @ basis_change
    assert np.linalg.cond(basis_change) < 3
    assert np.max(np.abs(transformed_0[:2, 2:])) < 2e-14
    assert np.max(np.abs(transformed_0[2:, :2])) < 2e-14

    unperturbed_blocks = [transformed_0[:2, :2], transformed_0[2:, 2:]]

    def solve_block_sylvester(rhs, index):
        if rhs is zero:
            return zero
        return solve_sylvester(
            unperturbed_blocks[index[0]],
            -unperturbed_blocks[index[1]],
            rhs,
        )

    effective, *_ = block_diagonalize(
        [transformed_0, transformed_1],
        subspace_indices=np.array([0] * 2 + [1] * 7),
        hermitian=False,
        solve_sylvester=solve_block_sylvester,
    )
    effective_order_2 = np.asarray(effective[0, 0, 2])
    effective_rates = np.sort(eigvals(effective_order_2).real)[::-1]
    np.testing.assert_allclose(effective_rates, [0, -20 / 3], atol=2e-14, rtol=0)

    exact_spectrum = eigvals(liouvillian_0 + omega_2 * liouvillian_1)
    exact_spectrum = exact_spectrum[np.argsort(-exact_spectrum.real)]
    exact_slow_rate = exact_spectrum[1].real
    exact_fast_rate = exact_spectrum[2].real
    predicted_slow_rate = omega_2**2 * effective_rates[1]
    relative_error = abs(predicted_slow_rate / exact_slow_rate - 1)
    separation = abs(exact_slow_rate / exact_fast_rate)
    assert relative_error < 0.12
    assert separation < 0.1

    print("Macieszczak et al. metastable three-level system")
    print(f"  exact slow rate: {exact_slow_rate:.6f}")
    print(f"  second-order slow rate: {predicted_slow_rate:.6f}")
    print(f"  slow/fast rate ratio: {separation:.4f}")


def kessler_third_order_superradiance():
    """Compare with Kessler's second- and third-order effective generators."""
    nuclear_dimension = 3
    nuclear_spin = 1
    number_of_spins = 2
    gamma = 1.0
    detuning = gamma / 5
    identity_nuclear = np.eye(nuclear_dimension, dtype=complex)
    electron_lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    electron_raising = electron_lowering.conj().T
    electron_occupation = electron_raising @ electron_lowering

    magnetic_numbers = np.arange(nuclear_spin, -nuclear_spin - 1, -1)
    lowering_coefficients = np.sqrt(
        (nuclear_spin + magnetic_numbers[:-1])
        * (nuclear_spin - magnetic_numbers[:-1] + 1)
    )
    collective_lowering = np.diag(lowering_coefficients, -1) / np.sqrt(number_of_spins)
    collective_raising = collective_lowering.conj().T
    collective_z = 2 * np.diag(magnetic_numbers) / np.sqrt(number_of_spins)

    hamiltonian_0 = detuning * kron(electron_occupation, identity_nuclear)
    liouvillian_0 = coherent_superoperator(hamiltonian_0)
    liouvillian_0 += dissipator_superoperator(
        np.sqrt(gamma) * kron(electron_lowering, identity_nuclear)
    )
    interaction = 0.5 * (
        kron(electron_raising, collective_lowering)
        + kron(electron_lowering, collective_raising)
    ) + kron(electron_occupation, collective_z)
    liouvillian_1 = coherent_superoperator(interaction)

    electron_ground_state = np.diag([1, 0])
    slow_vectors = []
    for row in range(nuclear_dimension):
        for column in range(nuclear_dimension):
            unit = np.zeros((nuclear_dimension, nuclear_dimension), dtype=complex)
            unit[row, column] = 1
            slow_vectors.append(kron(electron_ground_state, unit).reshape(-1))
    slow_basis = np.column_stack(slow_vectors)
    fast_basis = orth(liouvillian_0)
    basis = np.hstack((slow_basis, fast_basis))
    inverse_basis = np.linalg.inv(basis)
    transformed_0 = inverse_basis @ liouvillian_0 @ basis
    transformed_1 = inverse_basis @ liouvillian_1 @ basis
    slow_dimension = nuclear_dimension**2
    blocks = [
        transformed_0[:slow_dimension, :slow_dimension],
        transformed_0[slow_dimension:, slow_dimension:],
    ]

    def solve_block_sylvester(rhs, index):
        if rhs is zero:
            return zero
        return solve_sylvester(blocks[index[0]], -blocks[index[1]], rhs)

    effective, *_ = block_diagonalize(
        [transformed_0, transformed_1],
        subspace_indices=np.array(
            [0] * slow_dimension + [1] * (transformed_0.shape[0] - slow_dimension)
        ),
        hermitian=False,
        solve_sylvester=solve_block_sylvester,
    )
    pymablock_2 = np.asarray(effective[0, 0, 2])
    pymablock_3 = np.asarray(effective[0, 0, 3])

    denominator = (gamma / 2) ** 2 + detuning**2
    # Evaluating -V^- L0^-1 V^+ with the interaction g/2 gives a factor
    # four less than the rate printed below Eq. (23) of Kessler's paper.
    published_2 = (gamma / (4 * denominator)) * dissipator_superoperator(
        collective_lowering
    ).astype(complex)
    published_2 += coherent_superoperator(
        (-detuning / (4 * denominator)) * (collective_raising @ collective_lowering)
    )

    lambda_2 = -gamma / 2 + 1j * detuning
    lambda_3 = lambda_2.conjugate()
    lambda_4 = -gamma
    left = collective_lowering
    right = collective_raising
    z_left = collective_z @ collective_lowering
    right_z = collective_raising @ collective_z
    right_z_left = right_z @ collective_lowering
    published_3 = (
        1j
        / (4 * lambda_2**2)
        * (np.kron(left, right_z.T) - np.kron(identity_nuclear, right_z_left.T))
    )
    published_3 -= (
        1j
        / (4 * lambda_2 * lambda_4)
        * (np.kron(z_left, right.T) - np.kron(left, right_z.T))
    )
    published_3 += (
        1j
        / (4 * lambda_3**2)
        * (np.kron(right_z_left, identity_nuclear) - np.kron(z_left, right.T))
    )
    published_3 -= (
        1j
        / (4 * lambda_3 * lambda_4)
        * (np.kron(z_left, right.T) - np.kron(left, right_z.T))
    )

    errors = (
        np.max(np.abs(pymablock_2 - published_2)),
        np.max(np.abs(pymablock_3 - published_3)),
    )
    assert max(errors) < 2e-13
    for generator in (pymablock_2, pymablock_3):
        trace_residual, hermiticity_residual = structure_residuals(
            generator, nuclear_dimension
        )
        assert trace_residual < 1e-12
        assert hermiticity_residual < 1e-12

    print("Kessler third-order mediated superradiance")
    print(f"  second-order coefficient error: {errors[0]:.3e}")
    print(f"  third-order coefficient error: {errors[1]:.3e}")


def melo_variational_pt_gauge_comparison():
    """Relate Pymablock's stationary embedding to Melo et al.'s PT vectors."""
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_x = lowering + lowering.conj().T
    sigma_z = np.diag([1, -1])
    liouvillian_0 = coherent_superoperator(0.35 * sigma_z)
    liouvillian_0 += dissipator_superoperator(lowering)
    liouvillian_1 = coherent_superoperator(0.5 * sigma_x)
    steady_state = np.diag([1, 0]).reshape(-1).astype(complex)

    slow_basis = steady_state[:, None]
    fast_basis = orth(liouvillian_0)
    basis = np.hstack((slow_basis, fast_basis))
    inverse_basis = np.linalg.inv(basis)
    transformed_0 = inverse_basis @ liouvillian_0 @ basis
    transformed_1 = inverse_basis @ liouvillian_1 @ basis
    blocks = [transformed_0[:1, :1], transformed_0[1:, 1:]]

    def solve_block_sylvester(rhs, index):
        if rhs is zero:
            return zero
        return solve_sylvester(blocks[index[0]], -blocks[index[1]], rhs)

    effective, transformation, _ = block_diagonalize(
        [transformed_0, transformed_1],
        subspace_indices=np.array([0, 1, 1, 1]),
        hermitian=False,
        solve_sylvester=solve_block_sylvester,
    )
    max_order = 4
    diagonal = transformation[0, 0, 1 : max_order + 1]
    off_diagonal = transformation[1, 0, 1 : max_order + 1]
    embedding = [steady_state]
    for diagonal_order, off_diagonal_order in zip(diagonal, off_diagonal, strict=True):
        correction = np.zeros_like(steady_state)
        if diagonal_order is not np.ma.masked:
            correction += (slow_basis @ diagonal_order)[:, 0]
        if off_diagonal_order is not np.ma.masked:
            correction += (fast_basis @ off_diagonal_order)[:, 0]
        embedding.append(correction)
    assert np.all(effective[0, 0, : max_order + 1].mask)

    overlaps = np.array([np.vdot(steady_state, vector) for vector in embedding])
    inverse_overlap = np.zeros(max_order + 1, dtype=complex)
    inverse_overlap[0] = 1 / overlaps[0]
    for order in range(1, max_order + 1):
        inverse_overlap[order] = (
            -sum(
                overlaps[index] * inverse_overlap[order - index]
                for index in range(1, order + 1)
            )
            / overlaps[0]
        )
    pymablock_mp = [
        sum(
            embedding[index] * inverse_overlap[order - index]
            for index in range(order + 1)
        )
        for order in range(max_order + 1)
    ]

    pseudoinverse = np.linalg.pinv(liouvillian_0)
    melo_mp = [steady_state]
    for _ in range(max_order):
        melo_mp.append(-pseudoinverse @ liouvillian_1 @ melo_mp[-1])

    trace = np.eye(2).reshape(-1)
    source = np.array([1, 0, 0, 0], dtype=complex)
    trace_fixed = liouvillian_0 + np.outer(source, trace)
    lu_recurrence = [steady_state]
    for _ in range(max_order):
        raw = np.linalg.solve(trace_fixed, -liouvillian_1 @ lu_recurrence[-1])
        lu_recurrence.append(raw - np.vdot(steady_state, raw) * steady_state)

    mp_error = max(
        np.linalg.norm(pymablock_mp[order] - melo_mp[order])
        for order in range(max_order + 1)
    )
    lu_error = max(
        np.linalg.norm(lu_recurrence[order] - melo_mp[order])
        for order in range(max_order + 1)
    )
    assert mp_error < 2e-13
    assert lu_error < 2e-13

    print("Melo et al. steady-state perturbation gauge")
    print(f"  Pymablock versus Moore-Penrose recurrence: {mp_error:.3e}")
    print(f"  trace-fixed LU versus Moore-Penrose recurrence: {lu_error:.3e}")


def metastable_two_qubit_system():
    """Reproduce a four-dimensional two-qubit metastable manifold."""
    identity = np.eye(2, dtype=complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    sigma_x = lowering + raising
    occupation = raising @ lowering

    gamma_2 = 1.0
    gamma_1 = 4 * gamma_2
    omega_2 = gamma_2 / 50
    omega_1 = 2 * omega_2
    hamiltonian = omega_1 * kron(sigma_x, identity) + omega_2 * kron(identity, sigma_x)
    jump = np.sqrt(gamma_1) * kron(occupation, lowering) + np.sqrt(gamma_2) * kron(
        identity - occupation, raising
    )
    liouvillian = coherent_superoperator(hamiltonian) + dissipator_superoperator(jump)
    spectrum = eigvals(liouvillian)
    spectrum = spectrum[np.argsort(-spectrum.real)]

    slow = spectrum[:4]
    fast = spectrum[4:]
    separation = max(abs(slow.real)) / min(abs(fast.real))
    assert separation < 0.03
    assert np.max(np.abs(slow.imag)) < 1e-12

    print("Macieszczak et al. two-qubit metastability")
    print("  four slow eigenvalues:", np.array2string(slow, precision=6))
    print(f"  slow/fast real-part ratio: {separation:.4f}")


def tokieda_model(cutoff, thermal_occupation, detuning, gamma):
    """Construct the truncated oscillator-qubit model of Tokieda et al."""
    annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1).astype(complex)
    identity = np.eye(2, dtype=complex)
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T

    hamiltonian_0 = detuning * kron(annihilation.conj().T @ annihilation, identity)
    liouvillian_0 = coherent_superoperator(hamiltonian_0)
    liouvillian_0 += dissipator_superoperator(
        np.sqrt(gamma * (1 + thermal_occupation)) * kron(annihilation, identity)
    )
    liouvillian_0 += dissipator_superoperator(
        np.sqrt(gamma * thermal_occupation) * kron(annihilation.conj().T, identity)
    )
    interaction = kron(annihilation.conj().T, lowering) + kron(annihilation, raising)
    liouvillian_1 = coherent_superoperator(interaction)

    probabilities = (thermal_occupation / (1 + thermal_occupation)) ** np.arange(cutoff)
    oscillator_state = np.diag(probabilities / probabilities.sum())
    right_slow_vectors = []
    left_slow_vectors = []
    for row in range(2):
        for column in range(2):
            unit = np.zeros((2, 2), dtype=complex)
            unit[row, column] = 1
            right_slow_vectors.append(kron(oscillator_state, unit).reshape(-1))
            left_slow_vectors.append(kron(np.eye(cutoff), unit).reshape(-1))
    right_slow = np.column_stack(right_slow_vectors)
    left_slow = np.column_stack(left_slow_vectors)
    np.testing.assert_allclose(left_slow.conj().T @ right_slow, np.eye(4), atol=1e-14)
    return liouvillian_0, liouvillian_1, right_slow, left_slow


def tokieda_fourth_order_cp_obstruction():
    """Derive all fourth-order coefficients of Tokieda et al. with Pymablock."""
    gamma = 1.0
    thermal_occupation = 0.2
    lowering = np.array([[0, 1], [0, 0]], dtype=complex)
    raising = lowering.conj().T
    sigma_z = np.diag([-1, 1])
    templates = [
        coherent_superoperator(sigma_z / 2),
        dissipator_superoperator(lowering),
        dissipator_superoperator(raising),
        dissipator_superoperator(sigma_z),
    ]
    template_matrix = np.column_stack([template.reshape(-1) for template in templates])

    def published_coefficients(detuning):
        n_plus = thermal_occupation
        n_minus = 1 + thermal_occupation
        bar_gamma = gamma + 2j * detuning

        def b(order, occupation):
            if order == 2:
                return 2 * occupation / bar_gamma
            return 8 * occupation**2 / bar_gamma**3 + (
                8
                * n_plus
                * n_minus
                * (1 + 8j * gamma * detuning / abs(bar_gamma) ** 2)
                / (bar_gamma.conjugate() * abs(bar_gamma) ** 2)
            )

        b_minus_2, b_plus_2 = b(2, n_minus), b(2, n_plus)
        b_minus_4, b_plus_4 = b(4, n_minus), b(4, n_plus)
        x = 2 * detuning / gamma
        gamma_phi_4 = (
            -8 * n_plus * n_minus * (3 - 6 * x**2 - x**4) / (gamma**3 * (1 + x**2) ** 3)
        )
        return (
            np.array(
                [
                    (b_minus_2 + b_plus_2).imag,
                    2 * b_minus_2.real,
                    2 * b_plus_2.real,
                    0,
                ]
            ),
            np.array(
                [
                    (b_minus_4 + b_plus_4).imag,
                    2 * b_minus_4.real,
                    2 * b_plus_4.real,
                    gamma_phi_4,
                ]
            ),
        )

    def pymablock_coefficients(cutoff, detuning, return_embedding=False):
        liouvillian_0, liouvillian_1, right_slow, left_slow = tokieda_model(
            cutoff, thermal_occupation, detuning, gamma
        )
        effective, transformation, _ = block_diagonalize(
            [liouvillian_0, liouvillian_1],
            subspace_eigenvectors=[(right_slow, left_slow)],
            hermitian=False,
        )
        effective_2 = np.asarray(effective[0, 0, 2])
        effective_4 = np.asarray(effective[0, 0, 4])
        fitted_2, *_ = np.linalg.lstsq(
            template_matrix, effective_2.reshape(-1), rcond=None
        )
        fitted_4, *_ = np.linalg.lstsq(
            template_matrix, effective_4.reshape(-1), rcond=None
        )
        for effective_order, fitted in (
            (effective_2, fitted_2),
            (effective_4, fitted_4),
        ):
            fit_residual = np.linalg.norm(
                template_matrix @ fitted - effective_order.reshape(-1)
            )
            trace_residual, hermiticity_residual = structure_residuals(effective_order, 2)
            assert fit_residual < 1e-10
            assert trace_residual < 1e-10
            assert hermiticity_residual < 1e-10

        embedding_residual = None
        if return_embedding:
            diagonal = transformation[0, 0, 1:4]
            off_diagonal = transformation[1, 0, 1:4]
            embedding = [right_slow]
            for diagonal_order, off_diagonal_order in zip(
                diagonal, off_diagonal, strict=True
            ):
                correction = np.zeros_like(right_slow)
                if diagonal_order is not np.ma.masked:
                    correction += right_slow @ diagonal_order
                if off_diagonal_order is not np.ma.masked:
                    correction += off_diagonal_order
                embedding.append(correction)
            effective_orders = [
                np.zeros((4, 4), dtype=complex)
                if coefficient is np.ma.masked
                else coefficient
                for coefficient in effective[0, 0, :5]
            ]
            residuals = []
            for order in range(4):
                left = liouvillian_0 @ embedding[order]
                if order:
                    left += liouvillian_1 @ embedding[order - 1]
                right = sum(
                    embedding[index] @ effective_orders[order - index]
                    for index in range(order + 1)
                )
                residuals.append(np.linalg.norm(left - right))
            embedding_residual = max(residuals)
            assert embedding_residual < 2e-11
        return fitted_2.real, fitted_4.real, embedding_residual

    expected_2_zero, expected_4_zero = published_coefficients(0)
    fitted_by_cutoff = np.array(
        [pymablock_coefficients(cutoff, 0)[1] for cutoff in range(3, 7)]
    )
    errors = np.max(np.abs(fitted_by_cutoff - expected_4_zero), axis=1)
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 0.12

    detunings = np.array([0.0, 0.2, 0.4])
    detuning_results = []
    for detuning in detunings:
        expected_2, expected_4 = published_coefficients(detuning)
        fitted_2, fitted_4, _ = pymablock_coefficients(7, detuning)
        detuning_results.append((fitted_2, fitted_4, expected_2, expected_4))
    detuning_errors = np.array(
        [
            max(np.max(abs(fitted_2 - expected_2)), np.max(abs(fitted_4 - expected_4)))
            for fitted_2, fitted_4, expected_2, expected_4 in detuning_results
        ]
    )
    assert np.max(detuning_errors) < 0.08

    fitted_2_zero, fitted_4_zero, embedding_residual = pymablock_coefficients(
        6, 0, return_embedding=True
    )
    coefficient_lowering = np.array([1, -1j, 0]) / np.sqrt(2)
    coefficient_raising = coefficient_lowering.conj()
    coefficient_z = np.array([0, 0, np.sqrt(2)])

    def kossakowski(rates):
        return (
            rates[1] * np.outer(coefficient_lowering, coefficient_lowering.conj())
            + rates[2] * np.outer(coefficient_raising, coefficient_raising.conj())
            + rates[3] * np.outer(coefficient_z, coefficient_z)
        )

    kossakowski_2 = kossakowski(fitted_2_zero)
    kossakowski_4 = kossakowski(fitted_4_zero)
    kernel_vector = np.array([0, 0, 1], dtype=complex)
    assert np.linalg.norm(kossakowski_2 @ kernel_vector) < 1e-11
    kernel_certificate = np.vdot(kernel_vector, kossakowski_4 @ kernel_vector).real
    assert kernel_certificate < 0

    threshold = 0.5 * np.sqrt(-3 + 2 * np.sqrt(3))
    assert abs(threshold - 0.3406) < 1e-4
    assert fitted_by_cutoff[-1, 3] < 0
    assert detuning_results[-1][1][3] > 0

    print("Tokieda et al. fourth-order CP obstruction from Pymablock")
    print("  cutoffs 3..6 [omega, gamma_-, gamma_+, gamma_phi]:")
    for cutoff, fitted in zip(range(3, 7), fitted_by_cutoff, strict=True):
        print(f"    {cutoff}: {np.array2string(fitted, precision=5)}")
    print(f"  cutoff-6 maximum coefficient error: {errors[-1]:.3e}")
    print("  cutoff-7 maximum errors [Delta/gamma = 0, 0.2, 0.4]:")
    print(f"    {np.array2string(detuning_errors, precision=5)}")
    print(
        "  fitted gamma_phi signs:",
        np.sign([result[1][3] for result in detuning_results]).astype(int),
    )
    print(f"  C4 quadratic form on ker(C2): {kernel_certificate:.5f}")
    print(f"  embedding-equation residual: {embedding_residual:.3e}")
    print(f"  sign-change detuning |Delta|/gamma: {threshold:.4f}")


if __name__ == "__main__":
    reiter_sorensen_lambda_system()
    macieszczak_metastable_three_level_system()
    kessler_third_order_superradiance()
    melo_variational_pt_gauge_comparison()
    metastable_two_qubit_system()
    tokieda_fourth_order_cp_obstruction()
