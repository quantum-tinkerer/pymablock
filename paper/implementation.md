# Implementation

The main building block of all algorithms is a `BlockSeries` object, which
can refer to other series.
(2 blocks, infinitely many orders, they work with eval, compute when requested, cache)

The $U$ and $H_tilde$ series of Pymablock cross-reference each other using
`cauchy_dot_product`, so that they can be computed recursively.

In order to make Pymablock general, we only require the operators to support
addition and multiplication.
