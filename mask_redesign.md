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

## Proposed Solution: Matrix-Based Specification (Expression Encoded)

To avoid ambiguity, solutions that directly use `sympy.physics.quantum.Operator` objects (or SymPy expressions containing them) are preferred over string-based representations. **Operators and their daggers (e.g., `a` and `Dagger(a)`) will be treated distinctly.** If symmetric rules are desired (common in Hermitian problems), the user must specify rules for both the operator and its dagger explicitly (e.g., by providing `rules + rules.adjoint()` if `rules` is the matrix of expressions).

This approach uses a SymPy Matrix to define elimination rules, mapping rules directly to operator blocks.

- **Input**: A SymPy Matrix `M`, the same shape as the block operator.
- **Element `M[i, j]`**: A SymPy expression defining elimination rules for block `(i, j)`.
- **Semantics**:
  - **Default**: `0` or `None` in `M[i, j]` means keep all terms in that block.
  - **Exact Power**: `a**p` (where `p` is an integer) eliminates terms *exactly* matching `a**p` in block `(i, j)`. Does *not* eliminate `a**p * b` or `a**(p+1)`. Does *not* automatically apply to `Dagger(a)**p`.
  - **Threshold Power**: `a**(p + n)` (where `p` is an integer and `n` is a symbolic non-negative integer, e.g., `n = symbols('n', integer=True, nonnegative=True)`) eliminates terms containing `a` with power `k >= p` in block `(i, j)`. Does *not* automatically apply to `Dagger(a)`.
  - **Sums**: `expr1 + expr2` applies the elimination rules derived from *both* `expr1` and `expr2`. A term is eliminated if it matches *any* rule implied by the sum.
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
  mask_spec_matrix = Matrix([
      [0, a**3],  # Keep all in (0,0), eliminate terms exactly matching a**3 in (0,1)
      [ad**3 + bd**2, a**(2+n) + ad**(2+n)] # Eliminate exact ad**3 OR exact bd**2 in (1,0), eliminate a**k/ad**k for k>=2 in (1,1)
  ])
  ```

- **Pros**: Uses SymPy expressions directly. Matrix structure naturally maps rules to blocks. Symbolic exponents offer expressive power for thresholds. Explicit control over operator vs. dagger rules. Precise control over exact term matching vs. thresholds.
- **Cons**: Limited expressiveness for non-power rules (e.g., number conservation). Verbose if the same rule applies to many blocks, especially if symmetric rules are needed. Implementation complexity (parsing expressions, symbolic exponents).

**Challenges & Next Steps:**

This approach leverages SymPy expressions effectively for power-based rules and thresholds, but faces challenges:

- **Ambiguity**: The precise meaning of expressions (e.g., `a**3` - eliminate only this power or terms containing it?) needs strict definition.
- **Limited Scope**: Expressing non-power-based rules like number conservation seems difficult within this purely expression-based format.
- **Verbosity**: Applying the same rule across multiple blocks requires repetition. Explicitly defining rules for daggers increases verbosity for symmetric cases.

The immediate next steps involve resolving the remaining ambiguities:

- **Define Semantics**:
  - Clarify the default behavior (keep all or eliminate all?). **Decision: Keep all by default, matrix specifies eliminations.**
  - Specify if `a**p` eliminates only terms *exactly* matching `a**p` or any term *containing* `a**p`. **Decision: Exact match only.**
  - Define the interpretation of symbolic exponents like `a**(p + n)`. **Decision: Eliminate powers `>= p`.**
  - Define the interpretation of sums `expr1 + expr2`. **Decision: Apply elimination rules from `expr1` AND `expr2` (logical OR for elimination).**
- **Assess Expressiveness**: Can essential rules (like number conservation) be incorporated, perhaps via special symbols or functions within the expressions?
  - **Note on Number Conservation**: While not directly handled by simple power expressions, rules involving number conservation might be expressible later using products with shared symbolic exponents, e.g., `a**n * Dagger(b)**n` could potentially represent terms where the number change for `a` matches the number change for `b`. This requires further investigation into parsing and interpreting such expressions.
- **Plan Implementation**: Outline how to parse the matrix expressions and translate them into filtering logic, considering the defined semantics.
  - **Parsing**: For each element `M[i, j]` in the mask matrix, parse the SymPy expression using `NumberOrderedForm.from_expr(M[i, j], operators=...)`. This converts the rule expression into a `NumberOrderedForm` (`rule_nof`). The `operators` list should match the one used for the target operator being filtered.
  - **Filtering Logic**: Iterate through each term `(powers, coeff)` in the target `NumberOrderedForm` (`target_nof`) for block `(i, j)`.
    - Initialize `eliminate_term = False`.
    - For each term `(rule_powers, rule_coeff)` in `rule_nof`:
      - **Check Exact Match**: If `rule_coeff` is 1 (or equivalent simple scalar) and `powers == rule_powers`, set `eliminate_term = True` and break the inner loop.
      - **Check Threshold Match**: If `rule_coeff` involves a symbolic exponent `n` associated with a base power `p` for an operator `op` (e.g., derived from `op**(p+n)`), check if the corresponding power in `powers` is `>= p`. If yes, set `eliminate_term = True` and break the inner loop.
      - (Need to precisely define how to detect the threshold case from `rule_coeff` and `rule_powers`).
    - If `eliminate_term` is `True`, discard the `target_nof` term.
    - Otherwise, keep the `target_nof` term.
  - **Semantic Implementation**: The core challenge is translating the semantic decisions (especially threshold detection) into concrete checks within the filtering logic.
- **Future Interface Considerations**:
  - The `block_diagonalize` function should ideally accept *both* rules for terms to *eliminate* (as described by this matrix specification) and potentially rules for terms to *keep*. This would provide maximum flexibility for the user. The interaction between keep and eliminate rules needs careful definition (e.g., does elimination take precedence?).

**Considerations for Choosing a Solution:**

- **User Experience**: How intuitive and easy is it to write common filtering rules?
- **Expressiveness**: Can it handle all required scenarios?
- **Implementation**: How complex is the parsing and translation logic?
- **Performance**: Does the chosen format allow for efficient filtering?

## Other Options Considered (Discarded)

Several alternative approaches were considered before focusing on the Matrix-Based Specification:

1.  **String-Based Specification**: Using strings like `"a†"` or `"a†a"` to define rules. Ruled out due to potential ambiguity in parsing strings and mapping them reliably to specific operator objects.
2.  **Dictionary-Based Specification (Operator-Centric)**: Using base `Operator` objects as keys in a dictionary, with values defining rules (e.g., `keep_powers`, `keep_conserving`). This was a strong contender, especially with automatic operator/dagger symmetry. However, the matrix-based approach was chosen for its direct mapping of rules to blocks.
3.  **Simplified List/Rule Structure (Expression-Based)**: Using a list of dictionaries, each defining a rule with conditions (e.g., `term`, `condition`, `blocks`). Deemed potentially complex to implement the desired operator/dagger symmetry cleanly and required robust expression matching.
4.  **Function-Based Specification**: Allowing the user to provide a Python function `filter_func(...)` that returns `True` or `False` for each term. Offered maximum flexibility but placed a significant implementation burden on the user, was harder to serialize/validate, and potentially slower.
5.  **Leveraging `NumberOrderedForm` Shifts**: Defining rules based on the internal `shift` tuple representation within `NumberOrderedForm`. Less preferred as it exposed internal details and used string names as keys rather than operator objects directly in the specification.
