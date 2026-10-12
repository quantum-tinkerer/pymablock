## Quantitative convergence radii

The new bounds apply to the **actual Hermitian Pymablock series**, in a finite
complex matrix space, with any finite block partition and any finite number of
perturbation parameters. Let `delta > 0` be a lower bound on every cross-block
energy difference of the diagonal unperturbed Hamiltonian. All norms below are
Frobenius norms. Degeneracies inside a retained block are allowed.

### A simple bound to use directly

For `H(x) = H0 + sum_j x_j V_j`, a sufficient convergence region is

```text
sum_j |x_j| ||V_j||_F <= 9 delta / 130.
```

Equivalently, any closed polydisc with radii `r_j >= 0` satisfying
`sum_j r_j ||V_j||_F <= 9 delta / 130` has absolute convergence. In particular,
for a common radius and nonzero total perturbation norm,

```text
r_cert = 9 delta / (130 sum_j ||V_j||_F).
```

This has the expected linear dependence on the gap and inverse dependence on
perturbation strength. There is no additional matrix-size or block-count factor;
the Frobenius norms themselves can grow with system size. The coefficient `9/130`
is a conservative sufficient constant, not a claim about the optimal radius.

For a polynomial or analytic input, use the positive-degree majorant

```text
M(r) = sum_{n != 0} ||H_n||_F r^|n|.
```

Any radius with `M(r) <= 9 delta / 130` is certified. A polynomial makes this a
scalar polynomial inequality that can be solved directly. For an infinite
series with a known bound `M(R) <= M`, where `R > 0` and `M > 0`, an explicit
fallback is

```text
r_cert = R min(1, 9 delta / (130 M)).
```

It follows from `M(c R) <= c M(R)` for `0 <= c <= 1`, since the constant term
has been removed. Zero positive-degree mass means constant input and needs no
shrinkage. These bounds do not require a norm bound on `H0` itself.

Lean checks `gap_controls`, `matrix_radius_bound`, `affine_radius_certificate`,
and `analytic_radius_certificate`. The latter two certify the input bound to
feed into `matrix_radius_bound`. `blockDiagonalize_weighted` and `affine_mass`
cover nonnegative multiplicative weights, including unequal polydisc radii.
Both `U` and `H_tilde` converge at the specified radius; the theorem does not
silently replace it by an unspecified smaller radius.

### A sharper bound using the block structure

Let `D(r)` and `O(r)` be the weighted coefficient masses of the retained and
eliminated positive-degree input parts. A sufficient condition is

```text
7 O(r) + 6 D(r) <= 7 delta / 8.
```

For affine input, put `v_D = sum_j ||diag_blocks(V_j)||_F` and
`v_O = sum_j ||off_blocks(V_j)||_F`. This gives

```text
r_split = 7 delta / (8 (7 v_O + 6 v_D)).
```

Take the larger of this and the total-norm certificate. For purely off-block
input this improves to `r_split = delta / (8 v_O)`. An entirely retained
perturbation needs no transformation at all: `U = I`; these sufficient bounds
are not necessary conditions.

A stronger, still inexpensive certificate searches over one scalar `0 < t < 1/2`:

```text
O(r) (1 - 2 t^2) + 4 D(r) t (1-t)
    <= delta t (1-t/2) (1-2t).
```

For affine input, every such `t` certifies

```text
r(t) = delta t (1-t/2) (1-2t)
       / [v_O (1-2t^2) + 4 v_D t (1-t)].
```

Each candidate is a lower bound; finding the global optimum is unnecessary.
`solution_optimized_bound` proves this criterion, and
`blockDiagonalize_optimized_bound` transfers it to both returned series.

### Where the constants come from

Entrywise block projections are contractions in Frobenius norm, while the
Sylvester inverse has norm at most `1/delta`. Keeping these bounds separate
avoids combining dimensionless and inverse-energy constants.

If the coefficient masses of `q = U-I` and the auxiliary `B` are bounded by
`t` and `b`, one recurrence update obeys

```text
B_new <= 2 t b + (O + 2 D) t
q_new <= t^2/2 + [2 t b + O + (2 O + 4 D) t] / delta.
```

Choose `b = t (O+2D)/(1-2t)`. Substituting gives the scalar criterion above.
With `D,O <= epsilon`, `t=1/5` and `b=epsilon`, it reduces to
`epsilon <= 9 delta/130`. Choosing `t=1/4` gives the split certificate.
`step_majorant` and `solution_majorant` check the update bounds and pass them
through coefficient stabilization to the constructed formal solution.

### Exact two-level benchmark and limitations

Consider

```text
H(lambda) = diag(-1/2, 1/2) + lambda [[a,g],[g,-a]],  g != 0.
```

Its eigenvalue separation is `sqrt((1-2 a lambda)^2 + 4 g^2 lambda^2)`.
The nearest complex branch points are `lambda = 1/[2(a +/- i g)]`, so the
exact radius of the identity-branch unitary and transformed Hamiltonian is
`1/(2 sqrt(a^2+g^2))`. This analytic benchmark is an explanatory calculation,
not an additional Lean theorem. Real-axis absence of a crossing does not imply
an infinite Taylor radius.

| a | g | Total-norm bound | Split bound | Best grid certificate | Exact radius |
|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 0.048954 | 0.088388 | 0.088936 | 0.500000 |
| 1 | 1 | 0.034615 | 0.047594 | 0.048960 | 0.353553 |
| 4 | 1 | 0.011873 | 0.019959 | 0.022959 | 0.121268 |

Reproduce with `python tools/convergence_bounds.py`. The grid uses
`t=k/1000`, `1 <= k <= 499`, and checks every candidate against the original
coupled inequalities using exact rational arithmetic before printing decimals.
It does not estimate convergence from a finite list of perturbative coefficients.
The guaranteed radii are conservative, about 5.3 to 7.3 times smaller than the
exact radius for the optimized certificates in these examples.

There is no universal finite upper bound from the gap and perturbation norm
alone: a retained polynomial perturbation can give entire `U` and `H_tilde`.
Problem-specific complex singularities can supply upper bounds, as in the
benchmark. The present theorem does not locate them or prove optimality.
Non-Hermitian convergence remains outside its scope.

For context, [Bravyi, DiVincenzo and Loss, Section 3.4](https://arxiv.org/html/1105.0675#S3.SS4)
prove a two-subspace SW bound involving the gap, operator norm of the
perturbation, and width of the selected spectral interval. Their norm and
hypotheses differ from these Frobenius-norm, arbitrary-block certificates, so
the numerical constants should not be compared as if the bounds were identical.
