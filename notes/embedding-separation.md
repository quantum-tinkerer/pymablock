# Embeddings through the public NOF interface

The `feature/structured-embeddings` branch starts at `origin/main` (`ef1acf7`).
It contains the embedding API, occupation transitions, projected maps, recurrence
adapter, focused physical tests, and HTML explanation. It does not change
`number_ordered_form.py`, add packed storage, or change the ordinary
second-quantized Sylvester solver. The only change to `second_quantization.py`
is the public `Embedding` export.

The same feature is validated on `origin/packed-fermions` (`24b4034`) in
`test/structured-embeddings-packed`. The packed branch and the existing
`origin/optimize_bosons` branch remain separate features. No remote branch is
modified by this split.

## Representation boundary

Embedding code uses `NumberOrderedForm` construction, `from_expr`, `operators`,
`terms`, arithmetic, `adjoint`, and `as_expr`. It does not inspect packed keys,
SymPy `args`, private number placeholders, or private mode counts.

A source term is interpreted as an occupation shift and its coefficient. Its
bosonic amplitudes and fermionic signs follow from the declared operator types.
The coefficient coordinate associated with a mode is obtained by constructing
that mode's `NumberOperator` as a NOF and inspecting its sole diagonal term.
This uses the existing public interface without depending on placeholder names.

Two details matter when supporting both NOF implementations:

- A NOF can retain structurally zero coefficients. Map boundaries inspect term
  coefficients rather than assuming that truthiness always recognizes zero.
- Equal expressions can have different lists of unused generators. SymPy's
  adjoint cache may reuse one of these equivalent expressions. Map adjoints
  restore the declared generator list through public expression conversion.

Decoding `terms` may cost more than a specialized packed implementation. This
split establishes a common mathematical interface and correctness checks; it is
not a promise that the two storage implementations have identical performance.
A future fast transition iterator would belong to the NOF interface, with each
storage implementation supplying its own implementation behind that interface.

## Validation

The complete package suite passes with identical embedding source and tests:

| NOF implementation | Result | Elapsed time |
| --- | --- | --- |
| Current main (`ef1acf7`), unpacked | 407 passed, 1 expected failure | 143.23 s |
| Packed branch (`24b4034`) | 467 passed, 1 expected failure | 55.37 s |

These are individual local test runs, not a controlled performance benchmark.
The additional physical checks include the coupler's analytic mediated exchange
and fourth-order Hermiticity, the Hubbard-square coefficient `40 t**4 / U**3`,
the analytic first-order dressed spin-1 drive, and the spin-1 second-order matrix
compared against ordinary finite occupation matrices. The finite matrix oracle
uses no embedding transitions or NOF arithmetic to construct its reference.

The original combined branch is preserved. The large exploratory implementation
corpus stays there; the standalone feature carries only the required physical
fixtures and independent checks under `pymablock/tests/test_embedding_models.py`.
