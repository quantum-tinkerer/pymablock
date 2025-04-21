# Redesigning second quantization mask

## Problem statement

When specifying which operators to eliminate in the second quantized PT problems, pymablock uses a mask (see `pymablock.second_quantization.apply_mask_to_operator`), which is an extremely verbose format, nearly impossible for the user to specify.

In `pymablock.block_diagonalization.block_diagonalize` the mask is one of the inputs. While the internal format may possibly stay, the input format should definitely change to something more user-friendly.

## Current Implementation

The mask is currently defined as a type alias `Mask` in `pymablock.second_quantization`:

```python
Mask = tuple[
    list[sympy.physics.quantum.Operator],
    list[tuple[tuple[list[int], int | None], np.ndarray]],
]
```

As described in the `apply_mask_to_operator` docstring:

- The first element is a list of `Operator` objects whose powers will be checked.
- The second element is a list of selection rules.
- Each rule contains:
  - Constraints for each operator in the list (first element of `Mask`). A constraint is `(powers_to_keep: list[int], threshold: int | None)`. A term's power for an operator is kept if it's in `powers_to_keep` OR if `threshold` is not `None` and the power is `>= threshold`.
  - A boolean `np.ndarray` indicating which matrix elements `(i, j)` this rule applies to.

A term in the input `NumberOrderedForm` matrix element `operator[i, j]` is kept if:

1. The rule applies to the element `(i, j)` (checked via the boolean matrix).
2. The term's powers for *all* operators listed in the mask satisfy the corresponding constraints in that rule.
3. The term satisfies *at least one* such rule.

In `pymablock.block_diagonalization.block_diagonalize`, the `fully_diagonalize` parameter accepts this `Mask` type (usually within a dictionary mapping block indices to masks) when dealing with second-quantized operators (`operators` list is not empty) and fine-grained control over term elimination is needed beyond simple block diagonalization or numerical boolean masks. The primary interface seems geared towards simpler inputs (block indices or boolean arrays), but the complex `Mask` format is the underlying mechanism for detailed second-quantization term filtering.

The complexity arises from the nested tuples/lists and the need to manually construct boolean matrices for rule application across matrix elements.

## Design Plan: Matrix-Based Specification

To provide a more user-friendly way to specify term elimination rules, especially for second-quantized operators, we will implement a Matrix-Based Specification. This approach uses SymPy expressions within a matrix to define rules.

**Key Features:**

- **Direct Operator Use**: Leverages `sympy.physics.quantum.Operator` objects directly, avoiding string parsing ambiguity.
- **Distinct Operator/Dagger Treatment**: Operators and their daggers (e.g., `a` and `Dagger(a)`) are treated distinctly. Symmetric rules require explicit specification for both.
- **Block Mapping**: Uses a SymPy Matrix structure where the element `M[i, j]` defines rules specifically for the operator block `(i, j)`.

**Specification Details:**

- **Input**: A SymPy Matrix `M`, the same shape as the block operator being filtered.
- **Element `M[i, j]`**: A SymPy expression defining elimination rules for block `(i, j)`.
- **Semantics**:
  - **Default**: `0` in `M[i, j]` means keep all terms in that block.
  - **Exact Power**: `a**p` (where `p` is an integer) eliminates terms *exactly* matching `a**p` in block `(i, j)`. Does *not* eliminate `a**p * b` or `a**(p+1)`. Does *not* automatically apply to `Dagger(a)**p`.
  - **Threshold Power**: `a**(p + n)` (where `p` is an integer and `n` is a symbolic non-negative integer, e.g., `n = symbols('n', integer=True, nonnegative=True)`) eliminates terms containing `a` with power `k >= p` in block `(i, j)`. Does *not* automatically apply to `Dagger(a)`.
  - **Sums**: `expr1 + expr2` applies the elimination rules derived from *both* `expr1` and `expr2`. A term is eliminated if it matches *any* rule implied by the sum (logical OR for elimination).
- **Example**:

  ```python
  from sympy import Matrix, symbols
  from sympy.physics.quantum.boson import BosonOp
  from sympy.physics.quantum import Dagger
  a = BosonOp('a')
  b = BosonOp('b')
  ad = Dagger(a)
  bd = Dagger(b)
  n = symbols('n', integer=True, nonnegative=True)

  # For a 2x2 block operator
  elimination_rules = Matrix([
      [0, a**3],  # Keep all in (0,0), eliminate terms exactly matching a**3 in (0,1)
      [ad**3 + bd**2, a**(2+n) + ad**(2+n)] # Eliminate exact ad**3 OR exact bd**2 in (1,0), eliminate a**k/ad**k for k>=2 in (1,1)
  ])
  ```

**Implementation Plan:**

1. **Parsing**:
   - For each element `M[i, j]` in the input rule matrix `M`, parse the SymPy expression using `NumberOrderedForm.from_expr(M[i, j], operators=...)`. This converts the rule expression into a `NumberOrderedForm` (`rule_nof`).
   - The `operators` list provided to `from_expr` must match the one associated with the target operator being filtered.
2. **Filtering Logic**:
   - Create a new function, potentially `apply_expression_mask` or similar, that takes the target `NumberOrderedForm` (or matrix thereof) and the rule matrix `M`.
   - Iterate through each block `(i, j)` of the target operator.
   - Inside the block, iterate through each term `(powers, coeff)` in the target `NumberOrderedForm` (`target_nof`).
   - Retrieve the corresponding parsed rule `rule_nof` for block `(i, j)`.
   - Initialize `eliminate_term = False`.
   - Iterate through each rule term `(rule_powers, rule_coeff)` in `rule_nof`:
     - **Check Exact Match**: If `rule_coeff` is 1 (or equivalent simple scalar) and `powers == rule_powers`, set `eliminate_term = True` and break the inner loop.
     - **Check Threshold Match**: If `rule_coeff` indicates a threshold rule (e.g., derived from `op**(p+n)`), check if the corresponding power in `powers` meets the threshold (`>= p`). If yes, set `eliminate_term = True` and break the inner loop. (Requires a robust way to detect and interpret threshold rules from `rule_coeff` and `rule_powers`).
   - If `eliminate_term` is `False`, keep the `target_nof` term in the result for block `(i, j)`.
3. **Integration**:
   - Modify `block_diagonalize` to accept the rule matrix `M` (e.g., via the `fully_diagonalize` parameter).
   - Internally, call the new filtering function when this matrix format is provided.
4. **Testing**:
   - Implement unit tests for the parsing and filtering logic.
   - Test exact power elimination.
   - Test threshold power elimination.
   - Test elimination using sums of rules.
   - Test cases with `0` rules (keep all).
   - Test application across different blocks.

**Future Considerations:**

- **Number Conservation**: While not part of the initial implementation, investigate how rules like `a**n * Dagger(b)**n` could be parsed and interpreted to handle number conservation constraints.
- **Keep Rules**: Consider adding a parallel mechanism to specify terms to *keep*, potentially via another matrix argument to `block_diagonalize`. Define the precedence between keep and eliminate rules (e.g., elimination takes priority).

**Open Questions/Refinements:**

- **Threshold Detection**: The exact logic for detecting a threshold rule (`op**(p+n)`) from the parsed `rule_nof` needs refinement. How is the symbolic exponent `n` represented and identified after parsing?
