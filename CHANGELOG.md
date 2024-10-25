# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Generalized the algorithm to support an arbitrary number of blocks. To specify multiple blocks, provide either a list with eigenvectors of each block in `subspace_eigenvectors`, or a list marking to which block each basis state belongs in `subspace_indices`.
- Implemented full diagonalization of the Hamiltonian within blocks except for degenerate eigensubspaces. In case of one block with non-degenerate eigenvalues, this implements the Rayleigh-Schrödinger perturbation theory.
- Implemented selective diagonalization of the Hamiltonian within blocks, which can eliminate any subset of the off-diagonal elements within a block.

### Changed

- Auxiliary vectors for the implicit KPM solver should now be passed using `solver_options["aux_vectors"]` rather than as the last entry in `subspace_eigenvectors`.

### Improved

- Further reduced the number of matrix products by around 30% for high orders and down to a guaranteed minimum for 3rd order.
- Improved the efficiency of the MUMPS solver on real Hamiltonians.

### Fixed

- Fix incorrect shape of {autolink}`~pymablock.BlockSeries` blocks if $H_0$ has a zero block (#127).

## [2.0.0] - 2024-02-03

### Improved

- Switched to the `python-mumps` wrapper for the direct solver, which is available on all platforms and is more feature-complete.
- The implicit KPM solver now guarantees reaching a requested accuracy.
- A new algorithm that has optimal scaling while avoiding multiplication by $H_0$, and supports implicit data. This combines advantages of all previous algorithms, and therefore supersedes them.
- Sped up {autolink}`~pymablock.series.cauchy_dot_product` when there are more than 3 series by reusing intermediate results.
- Optimized memory usage of `~pymablock.block_diagonalize` by deleting intermediate results when they are no longer needed.

### Added

- A complete description of the algorithm to the documentation, see [documentation](algorithms.md).
- String representation of {autolink}`~pymablock.BlockSeries` for readability.

### Removed

- `expanded`, `symbolic`, `implicit`, and `general` functions and algorithms (functionality taken over by the new general algorithm, with {autolink}`~pymablock.block_diagonalize` the main interface).
- the `algorithm` argument from {autolink}`~pymablock.block_diagonalize` (there is only one algorithm now).
- `exclude_last` argument of {autolink}`~pymablock.series.cauchy_dot_product` (instead we check whtether other terms lack 0th order).

## [1.0.0] - 2023-06-05

- First release of _Pymablock_.
