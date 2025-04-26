# Second Quantization Tools

This document explains the conceptual foundations of the second quantization tools in Pymablock.
Specifically, it covers the implementation of number-ordered forms and the solution of Sylvester equations for quantum operators in second quantization, which are the basis for Pymablock's approach to quantum mechanical calculations.

## Number-Ordered Forms

### Concept and Significance

#### From Normal Ordering to Number-Ordered Forms

In many-body physics, normal ordering is a fundamental concept where creation operators are placed to the left of annihilation operators in an expression: $a^\dagger a a^\dagger$ becomes $a^\dagger a^\dagger a + a^\dagger$ when normal-ordered.
Normal ordering provides a reference for calculating expectation values and is essential for perturbation theory.

However, simplifying complex expressions with normal ordering is challenging, because the non-commutative nature of operators leads to an explosion of number of terms as expressions grow.
Even worse, non-polynomial functions of operators—such as when number operators appear in denominators (like $(a^\dagger a + 2)^{-1}$)—cannot be represented in normal-ordered form at all.

Instead, Pymablock uses a **number-ordered form** to represent second-quantized operators.
Number-ordered form satisfies the key property that **no term can simultaneously contain both creation and annihilation operators for the same quantum mode**.
In addition to this property, number-ordered form specifies a strict ordering of operators in each term:

1. Sorted creation operators appear on the left
2. Annihilation operators appear on the right in reverse sorted order
3. Number operators $N_a \equiv a^\dagger a$ and scalar expressions appear in the middle (called the "coefficient" of the term)

This representation is motivated by making the largest part of the expression commutative, and forming as many number operators as possible.
For example, because the coefficients only contain commuting operators, they can be simplified using standard algebraic tools.
This form also allows to represent and simplify non-polynomial functions of number operators.

### Mathematical Structure and Implementation

A number-ordered form is a sum of terms, where each term consists of:

- A set of creation operators for various modes
- A set of annihilation operators for different modes (never the same modes as the creation operators)
- A coefficient that may contain number operators and scalar values

In the implementation, these concepts are represented as:

```python
# A number-ordered form is represented with:
class NumberOrderedForm:
    # 1. A sorted list of all operators (modes)
    operators = [a, b, c, ...]

    # 2. A dictionary mapping operator powers to coefficients
    terms = {
        (a_power, b_power, c_power, ...): coefficient_expression
    }
```

For example, consider the expression $a^\dagger b + 2$ (where $a$ and $b$ are different modes):

Mathematically, this represents:

- The term $a^\dagger b$ with one creation operator for mode $a$ and one annihilation operator for mode $b$
- A scalar term 2

In the implementation, this would be represented as:

```python
operators = [a, b]
terms = {
    (-1, 1): 1,  # a† b with coefficient 1
    (0, 0): 2    # constant term 2
}
```

where negative values in the tuple indicate creation operators and positive values indicate annihilation operators.

A more complex expression $2 \cdot a^{\dagger 2} \cdot N_b \cdot c^3 + a^{\dagger} \cdot [3N_a/(N_b+1)] \cdot c^2$ is represented as:

Mathematically:

- First term: Two creation operators for mode $a$, a number operator for mode $b$ as part of the coefficient, three annihilation operators for mode $c$, and a scalar coefficient of 2
- Second term: One creation operator for mode $a$, two annihilation operators for mode $c$, and a non-polynomial coefficient $(3N_a)/(N_b+1)$

In the implementation:

```python
operators = [a, b, c]
terms = {
    (-2, 0, 3): 2 * N_b,          # a†² c³ with coefficient 2*N_b
    (-1, 0, 2): 3*N_a/(N_b+1)     # a† c² with non-polynomial coefficient
}
```

For more details about the {autolink}`~pymablock.number_ordered_form.NumberOrderedForm` class, see the documentation.

### Quantum Operator Multiplication

The real power of number-ordered forms becomes apparent when we multiply quantum operators.
The multiplication of of quantum operators follows from the commutation relations of individual operators.

For example, the bosonic commutation relation forms the foundation of all manipulations with bosons:

$$[a, a^\dagger] = aa^\dagger - a^\dagger a = 1 \quad \Rightarrow \quad aa^\dagger = 1 + a^\dagger a = 1 + N_a.$$

This leads directly to the rule for how operators shift number operators:

$$a \cdot f(N_a) = f(N_a - 1) \cdot a \quad\quad a^\dagger \cdot f(N_a) = f(N_a + 1) \cdot a^\dagger.$$

To illustrate how number-ordered form manipulations work in practice, let's examine multiplication rules for a single mode through a multiplication table. This table shows the result of multiplying different terms of a number-ordered form of an operator from the left by various operators.

Consider three possible forms of number-ordered terms with a single mode:

1. $(a^\dagger)^n \cdot f(N_a)$ - Creation operators followed by a function of the number operator
2. $g(N_a)$ - Just a function of the number operator
3. $h(N_a) \cdot a^m$ - A function of the number operator followed by annihilation operators

The table below shows what happens when we multiply these terms (columns) from the left by different operators (rows):

| Left × Term | $(a^\dagger)^n \cdot f(N_a)$ | $g(N_a)$ | $h(N_a) \cdot a^m$ |
|-------------|------------------------------|----------|-------------------|
| $a^\dagger$ | $(a^\dagger)^{n+1} \cdot f(N_a)$ | $a^\dagger \cdot g(N_a)$ | $h(N_a+1) N_a \cdot a^{m-1}$ |
| $j(N_a)$ | $(a^\dagger)^n \cdot j(N_a+n) \cdot f(N_a)$ | $j(N_a) \cdot g(N_a)$ | $j(N_a) \cdot h(N_a) \cdot a^m$ |
| $a$ | $(a^\dagger)^{n-1} \cdot (N_a - n + 2) \cdot f(N_a)$ | $g(N_a-1) \cdot a$ | $h(N_a-1) \cdot a^{m+1}$ |

This multiplication table provides a systematic way to derive any number-ordered term by multiplying a number-ordered term from the left with any operator.
By repeatedly applying these rules to all terms in a number-ordered expression, we compute the product of any sequence of operators while maintaining the number-ordered form.

The algebraic framework is completed with two additional operations.
First, addition simply merges coefficients of terms with identical operator powers.
Second, taking the adjoint negates all the powers, which turns creation operators into annihilation operators and vice versa, and conjugates the coefficients.
Together, these operations provide all the necessary tools to manipulate quantum expressions in number-ordered form in Pymablock.

### Use within Pymablock

Whenever the user provides input to {autolink}`~pymablock.block_diagonalize` which is a matrix containing second quantized operators, Pymablock computes the output as {autolink}`~pymablock.series.BlockSeries` of matrices containing number-ordered forms.
Furthermore, all coefficients of the number-ordered forms are expressions containing {autolink}`~pymablock.number_ordered_form.NumberOperator` objects, which Pymablock uses to avoid storing the original $a^\dagger a$ terms in the coefficients.

To convert the matrices to regular sympy expressions, use `result.applyfunc(lambda x: x.as_expr())`, however for many applications this is not necessary because the number-ordered forms already allow to easily manipulate the expressions.

Then to replace the number operators with their operator form, use `result.doit()` to obtain the matrix with the original creation and annihilation operators.

## Solving Sylvester Equations

Once per perturbative order, Pymablock solves a Sylvester equation as described in the [algorithm](algorithms.md).
The quantum Sylvester equation takes the form:

$$H_i X_{ij} - X_{ij} H_j = Y_{ij}$$

where $H_i$ and $H_j$ are Hamiltonian blocks, $Y_{ij}$ is a perturbation term, and $X_{ij}$ is the unknown.

Let us solve this equation when the Hamiltonian only contains a single mode $a$.

$$H_i X - X H_j = Y$$

$H_i$ and $H_j$ are functions of $N_a$.
$Y$ is a number-ordered term with either creation or annihilation operators (not both) for a given mode.

We express $Y = (a^\dagger)^n \cdot f_Y(N) \cdot a^m$ (with either $n=0$ or $m=0$).
We then use an Ansatz for $X$ with the same operator structure: $X = (a^\dagger)^n \cdot f_X(N) \cdot a^m$, so that the Sylvester equation becomes:

$$H_i(N) \cdot (a^\dagger)^n \cdot f_X(N) \cdot a^m - (a^\dagger)^n \cdot f_X(N) \cdot a^m \cdot H_j(N) = (a^\dagger)^n \cdot f_Y(N) \cdot a^m.$$

We then commute $H_i$ and $H_j$ into the middle $H_i(N) \cdot (a^\dagger)^n = (a^\dagger)^n \cdot H_i(N-n)$ and $a^m \cdot H_j(N) = H_j(N+m) \cdot a^m$, which transforms the equation to:

$$(a^\dagger)^n \cdot [H_i(N-n) \cdot f_X(N) - f_X(N) \cdot H_j(N+m)] \cdot a^m = (a^\dagger)^n \cdot f_Y(N) \cdot a^m.$$

Because $H_i$ and $H_j$ commute with $f_X(N)$, the solution is:

$$X = (a^\dagger)^n \cdot \frac{f_Y(N)}{H_i(N-n) - H_j(N+m)} \cdot a^m$$

The generalization to multiple modes follows the same pattern: for each mode, apply the appropriate shifts to the Hamiltonian based on the creation and annihilation operators in the perturbation term and find the solution with the same powers of creation and annihilation operators as the right hand side.

In Pymablock, the {autolink}`~pymablock.second_quantization.solve_sylvester_bosonic` function implements this approach.

## Filtering terms of number ordered forms

When working with second quantized operators in perturbation theory, the goal is often to eliminate specific terms from the Hamiltonian.
Pymablock provides a format for specifying which terms to keep or eliminate based on the powers of creation and annihilation operators.

Which operators to eliminate is defined by:

- An expression matrix `M`, which has the same shape as the operator being filtered
- Each element `M[i, j]` defines the elimination rules for element `(i, j)` of the operator: every term of the number-ordered form of that element is eliminated from the corresponding element of the operator.

### Example

For a 2×2 block operator with modes `a` and `b`, you can specify:

```python
from sympy import Matrix, symbols
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger

a = BosonOp('a')
b = BosonOp('b')
n = symbols('n', integer=True, nonnegative=True)

# Define elimination rules for each block
elimination_rules = Matrix([
    [0, a**3],                             # Keep all in (0,0), eliminate a³ in (0,1)
    [Dagger(a)**3 + Dagger(b)**2, a**(2+n) + Dagger(a)**(2+n)]  # Multiple rules in (1,0) and (1,1)
])
```

This specifies:

- Block (0,0): Keep all terms
- Block (0,1): Eliminate terms with exactly 3 annihilation operators for mode `a`
- Block (1,0): Eliminate terms with either 3 creation operators for mode `a` or 2 creation operators for mode `b`
- Block (1,1): Eliminate terms with 2 or more creation or annihilation operators for mode `a`

This functionality is implemented by {autolink}`~pymablock.second_quantization.apply_mask_to_operator`.

### Application in Block Diagonalization

When using {autolink}`~pymablock.block_diagonalize` with second quantized operators, the `fully_diagonalize` parameter accepts this matrix-based format to control term elimination:

```python
H_tilde, *_ = block_diagonalize(
    sympy.Matrix([[H_0 + H_p]]),
    fully_diagonalize=elimination_rules,
    symbols=[g]
)
```

This approach provides fine-grained control over which terms to include in the effective Hamiltonian, making it possible to implement physical approximations like the rotating wave approximation or selective truncation of higher-order terms.
