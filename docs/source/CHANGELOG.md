# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Improved
- A new algorithm that has optimal scaling while avoiding multiplication by
  $H_0$, and supports implicit data. This combines advantages
  of all previous algorithms, and therefore supersedes them.
- Sped up `~pymablock.series.cauchy_dot_product` when there are more than 3 series by reusing intermediate results.

### Added
- A complete description of the algorithm to the documentation, see
  [documentation](algorithms.md).
- String representation of `~pymablock.BlockSeries` for readability.

### Removed
- `pymablock.expanded`, `pymablock.symbolic`, `pymablock.implicit`, and
  `pymablock.general` functions and algorithms (functionality taken over by the
  new general algorithm, with `pymablock.block_diagonalize` the main interface).
- the `algorithm` argument from `~pymablock.block_diagonalize` (there is only one algorithm now).
- `exclude_last` argument of `~pymablock.series.cauchy_dot_product` (instead we check whtether other terms lack 0th order).

## [1.0.0] - 2023-06-05

- First release of _Pymablock_.
