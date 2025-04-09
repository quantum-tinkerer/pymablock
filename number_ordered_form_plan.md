# Plan for Implementing NumberOrderedForm as an Operator Class

## Motivation
The current implementation of number ordered form in `second_quantization.py` is convenient for computation but lacks an explicit format. Extracting data from it requires auxiliary functions that are error-prone. We can improve this by implementing a dedicated `NumberOrderedForm` class as a subclass of `sympy.physics.quantum.Operator`.

## Number Ordered Form Definition
A number ordered form is a representation of quantum operators where:

1. All creation operators are on the left
2. All annihilation operators are on the right
3. Number operators (and other scalar expressions) are in the middle

The key property of this ordering is that **no term in a number-ordered expression can simultaneously contain both a creation and an annihilation operator for the same particle**. This form makes it easy to manipulate complex quantum expressions, because commuting a creation or annihilation operator through a function of a number operator simply replaces the corresponding number operator `N` with `N ± 1`.

The current implementation in `number_ordered_form()` uses the following approach:
- Processes expressions term by term
- For each term, applies multiplication rules to maintain the ordered form
- Groups similar terms using `group_ordered()` which sorts by creation and annihilation operator patterns
- Can optionally simplify the number operator expressions

### Data Structure
The `NumberOrderedForm` class will store:

- **Operators list**: A sorted list of all operators (bosonic and fermionic) involved in the expression, with bosons sorted before fermions
- **Terms dictionary**: A dictionary where:
  - **Keys**: Integer tuples representing the powers of operators (negative powers for creation operators, positive powers for annihilation operators)
  - **Values**: Sympy expressions containing only number operators and coefficients that multiply the corresponding operator term

For example, a term like `2 * a†^2 * n_b * a^3` would be represented with:
- Key: (-2, 3) for the operator a (note: creation operator powers are negative in our implementation)
- Value: 2 * n_b (the coefficient and number operators)

This structure explicitly shows which creation and annihilation operators are present in each term, while clearly separating the number operators and coefficients. It also makes it easy to apply commutation rules and perform algebraic operations.

## Existing Functions to Integrate

The current implementation in `second_quantization.py` has several functions related to number ordered form that will need to be integrated or adapted. Many of these will be completely replaced by the direct structure of the `NumberOrderedForm` class:

1. **`number_ordered_form(expr, simplify=False)`** - Will be replaced by the `NumberOrderedForm` constructor, though it may remain as a factory function that returns a `NumberOrderedForm` instance

2. **`multiply_b(expr, operator)`**, **`multiply_daggered_b(expr, daggered_operator)`**, and **`multiply_fn(expr, nexpr)`** - These will be replaced by overloading multiplication operators in the `NumberOrderedForm` class. When multiplying two `NumberOrderedForm` objects or a `NumberOrderedForm` with another operator, the class will automatically maintain the number ordered form in the result.

5. **`group_ordered(expr)`** - Will be completely eliminated as the `NumberOrderedForm` data structure is inherently grouped by operator powers

6. **`simplify_number_expression(expr)`** - Will become a method of `NumberOrderedForm` called `simplify()` that operates on the internal representation

7. **`expr_to_shifts(expr, boson_operators)`** - Will be entirely replaced as the `NumberOrderedForm` data structure directly represents shifts in its dictionary keys

8. **`find_operators(expr)`** - Will be used by the `NumberOrderedForm` constructor to identify all operators in the expression

9. **`solve_monomial(Y, H_ii, H_jj, boson_operators)`** - Will need to be updated to accept and return `NumberOrderedForm` instances instead of raw expressions

10. **`solve_sylvester_bosonic(...)`** - Will need to be updated to work with `NumberOrderedForm` objects but will remain a separate function

11. **`apply_mask_to_operator(operator, mask)`** - Will be implemented as a method on `NumberOrderedForm` that directly operates on its internal data structure, making filtering much more efficient

## Test Plan

Based on the existing tests in the codebase and the planned functionality of the `NumberOrderedForm` class, we should test the following aspects:

### Construction and Conversion Tests
- [ ] **Basic Construction** - Test creating `NumberOrderedForm` instances from various expressions
- [ ] **SymPy Protocol Compliance** - Test that `obj == type(obj)(*obj.args)` holds true for our class (essential for SymPy compatibility)
- [ ] **Round-Trip Conversion** - Test that converting to `NumberOrderedForm` and back to a standard sympy expression preserves equality
- [ ] **Error Handling** - Test proper error handling for invalid inputs (e.g., negative powers of boson operators)
- [ ] **Operator Detection** - Test that all operators are properly identified and sorted (bosons before fermions)

### Mathematical Operation Tests
- [ ] **Addition** - Test adding two `NumberOrderedForm` instances
- [ ] **Multiplication** - Test multiplying `NumberOrderedForm` with various types:
  - [ ] With another `NumberOrderedForm`
  - [ ] With creation operators
  - [ ] With annihilation operators
  - [ ] With number operators
  - [ ] With scalar expressions
- [ ] **Commutation Relations** - Test commutation with different operator types
- [ ] **Power Operations** - Test raising `NumberOrderedForm` to powers

### Utility Method Tests
- [ ] **Simplification** - Test that `simplify()` properly simplifies number operator expressions
- [ ] **Component Extraction** - Test extracting creation, annihilation, and number parts
- [ ] **Mask Application** - Test filtering terms based on operator powers
- [ ] **Algebraic Properties** - Test for properties like associativity, distributivity, etc.

### Compatibility Tests
- [ ] **Integration with Sylvester Equation Solver** - Test that existing solver functions work with `NumberOrderedForm`
- [ ] **Hermiticity Preservation** - Test that operations preserve hermiticity where expected
- [ ] **Block Diagonalization** - Test integration with the block diagonalization functionality

### Edge Cases and Special Properties
- [ ] **Empty Expressions** - Test behavior with zero or empty expressions
- [ ] **Idempotence** - Test that ordering an already number-ordered form leaves it unchanged
- [ ] **Complex Expressions** - Test with complex expressions containing multiple operators and terms
- [ ] **Performance** - Benchmark against current implementation for speed and memory usage

Most of these tests can be modeled after existing tests in `test_second_quantization.py`, particularly `test_number_ordered_form`, `test_group_ordered_idempotence`, and `test_multiply_*` functions, adapting them to work with the new class-based approach.

## Implementation Plan

- [ ] **1. Design and implement the NumberOrderedForm class initialization**
  - [ ] Subclass from `sympy.physics.quantum.Operator` to integrate with sympy's operator system
  - [ ] Implement the core data structure:
    - [ ] `operators`: Sorted list of quantum operators (bosons before fermions)
    - [ ] `terms`: Dictionary mapping integer tuples (operator powers) to coefficient expressions
  - [ ] Create basic constructor that only accepts direct arguments:
    - [ ] `operators`: List of quantum operators
    - [ ] `terms`: Dictionary with operator powers as keys and coefficients as values

- [ ] **2. Implement validation methods**
  - [ ] Validate operators list (must be valid quantum operators, bosons before fermions)
  - [ ] Validate terms dictionary structure (keys must be tuples matching operators length, values must be valid expressions)
  - [ ] Verify number operators in the coefficient expressions are consistent with the provided operators
  - [ ] Check for fermion operators to ensure proper handling of anti-commutation

- [ ] **3. Implement key operator overloads**
  - [ ] Addition (`__add__`): Combine terms with the same operator powers
  - [ ] Multiplication (`__mul__`): Handle all cases (NumberOrderedForm × NumberOrderedForm, NumberOrderedForm × operator, etc.)
  - [ ] Commutation (`_eval_commutator_*`): Implement commutation rules with various operator types
  - [ ] Powers, negation and other algebraic operations

- [ ] **4. Implement direct manipulation methods**
  - [ ] `get_creation_part()`: Extract only creation operator components
  - [ ] `get_annihilation_part()`: Extract only annihilation operator components
  - [ ] `get_number_part()`: Extract only number operator components
  - [ ] `filter_powers(powers_dict)`: Filter terms based on operator powers

- [ ] **5. Implement utility methods**
  - [ ] `simplify()`: Simplify the number operator expressions in the terms
  - [ ] `apply_mask(mask)`: Apply filtering mask (replacing current `apply_mask_to_operator`)
  - [ ] `get_shifts()`: Get all the operator power shifts in the expression
  - [ ] `sort()`: Normalize the internal representation

- [ ] **6. Implement display methods**
  - [ ] String representation (`__str__`) for readable console output
  - [ ] LaTeX representation (`_latex_`) for mathematical typesetting
  - [ ] Pretty printing for nicer terminal output

- [ ] **7. Update existing functions to use NumberOrderedForm**
  - [ ] Convert `number_ordered_form()` to a factory function returning a NumberOrderedForm instance
  - [ ] Update `solve_monomial()` and `solve_sylvester_bosonic()` to accept and return NumberOrderedForm
  - [ ] Replace `apply_mask_to_operator()` with the class method
  - [ ] Update any code that relies on the current implementation pattern

- [ ] **8. Comprehensive testing**
  - [ ] Test initialization from various expressions (bosonic, fermionic, mixed)
  - [ ] Test algebraic operations (addition, multiplication, powers)
  - [ ] Test commutation relations with different operator types
  - [ ] Test utility methods like simplify and apply_mask
  - [ ] Test compatibility with existing codebase functionality
  - [ ] Benchmark against current implementation

- [ ] **9. Documentation**
  - [ ] Comprehensive class docstring explaining the data structure and approach
  - [ ] Method docstrings with parameters, return types, and examples
  - [ ] Update module-level documentation to explain the new class
  - [ ] Add usage examples in docstrings and tests

## Benefits
- More explicit representation of number-ordered expressions
- Easier extraction of components
- More robust algebraic operations
- Better error messages
- Cleaner code with less auxiliary functions
