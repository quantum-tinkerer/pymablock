# Second Quantization Tools

This document explains the conceptual foundations of the second quantization tools in Pymablock.
Specifically, it covers the implementation of number-ordered forms and the solution of Sylvester equations for quantum operators in second quantization, which are the basis for Pymablock's approach to quantum mechanical calculations.

## Number-Ordered Forms

### Concept and Significance

#### From Normal Ordering to Number-Ordered Forms

In quantum field theory and many-body physics, normal ordering is a fundamental concept where creation operators are placed to the left of annihilation operators in an expression: $a^\dagger a a^\dagger$ becomes $a^\dagger a^\dagger a + a^\dagger$ when normal ordered.
Normal ordering provides a reference for calculating expectation values and is essential for perturbation theory.

However, simplifying expressions with normal ordering is challenging for complex quantum systems, because the non-commutative nature of quantum operators leads to an explosion of terms as expressions grow.
Even worse, non-polynomial functions of operators—such as when number operators appear in denominators (like $\frac{1}{N_a + 2}$)—cannot be represented in normal-ordered form at all.

Instead, Pymablock uses a **number-ordered form** to represent second quantized operators.
The number-ordered form addresses these challenges by providing a more structured representation where:

1. Creation operators appear on the left
2. Annihilation operators appear on the right
3. Number operators and scalar expressions appear in the middle

The fundamental property of this ordering is that **no term can simultaneously contain both creation and annihilation operators for the same quantum mode**.
This separation creates a clean structure with a crucial advantage: the middle part of each term contains only commuting operators (number operators and scalars), which can be simplified using standard algebraic tools.
This allows to represent and simplify non-polynomial functions of number operators, for example.
Altogether, number-ordered forms are a powerful tool for manipulating second quantized operators in Pymablock.

### Mathematical Structure and Implementation

At its core, a number-ordered form is a sum of terms, where each term consists of:

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

For a more complex example, the expression $2 \cdot a^{\dagger 2} \cdot n_b \cdot c^3 + a^{\dagger} \cdot \frac{3N_a}{N_b+1} \cdot c^2$ represents:

Mathematically:

- First term: Two creation operators for mode $a$, a number operator for mode $b$ as part of the coefficient, three annihilation operators for mode $c$, and a scalar coefficient of 2
- Second term: One creation operator for mode $a$, two annihilation operators for mode $c$, and a non-polynomial coefficient $\frac{3N_a}{N_b+1}$

In the implementation:

```python
operators = [a, b, c]
terms = {
    (-2, 0, 3): 2 * n_b,          # a†² c³ with coefficient 2*n_b
    (-1, 0, 2): 3*N_a/(N_b+1)     # a† c² with non-polynomial coefficient
}
```

For more details about the `pymablock.number_ordered_form.NumberOrderedForm` class, see the documentation.

### Quantum Operator Multiplication

The real power of number-ordered forms becomes apparent when we multiply quantum operators.
When creation and annihilation operators appear in an expression, they follow specific commutation rules.

For example, the bosonic commutation relation forms the foundation of all manipulations with bosons:

$$[a, a^\dagger] = aa^\dagger - a^\dagger a = 1 \quad \Rightarrow \quad aa^\dagger = 1 + a^\dagger a = 1 + N_a$$

This leads directly to the rule for how operators shift number operators:

$$a \cdot f(N_a) = f(N_a - 1) \cdot a \quad\quad a^\dagger \cdot f(N_a) = f(N_a + 1) \cdot a^\dagger$$

To illustrate how number-ordered form manipulations work in practice, let's examine multiplication rules for a single mode through a multiplication table. This table shows the result of multiplying different forms of number-ordered terms from the left by various operators.

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

This multiplication table provides a systematic way to derive any number-ordered term by multiplying a number-ordered term from the left with an operator.
By repeatedly applying these rules for all terms in a number-ordered expression, we compute the product of any sequence of operators while maintaining the number-ordered form.

The algebraic framework is completed with two additional operations.
First, addition simply merges terms with identical operator powers.
Second, taking the adjoint negates all the powers, which turns creation operators into annihilation operators and vice versa, and conjugates the coefficients.
Together, these operations provide all the necessary tools to manipulate quantum expressions in number-ordered form in Pymablock.

## Solving Sylvester Equations

Once per perturbative order, Pymablock solves a Sylvester equation as described in the [algorithm](algorithms.md).
The quantum Sylvester equation takes the form:

$$H_i X_{ij} - X_{ij} H_j = Y_{ij}$$

where $H_i$ and $H_j$ are Hamiltonian blocks, $Y_{ij}$ is a perturbation term, and $X_{ij}$ is the unknown.

The number-ordered approach solves the Sylvester equation for a single quantum mode:

$$H_i X - X H_j = Y$$

$H_i$ and $H_j$ are functions of $N$.
$Y$ is a number-ordered term with either creation or annihilation operators (not both) for a given mode.

Express $Y = (a^\dagger)^n \cdot f_Y(N) \cdot a^m$ (with either $n=0$ or $m=0$).
$X$ shares this operator structure: $X = (a^\dagger)^n \cdot f_X(N) \cdot a^m$. The Sylvester equation becomes:

$$H_i(N) \cdot (a^\dagger)^n \cdot f_X(N) \cdot a^m - (a^\dagger)^n \cdot f_X(N) \cdot a^m \cdot H_j(N) = (a^\dagger)^n \cdot f_Y(N) \cdot a^m$$

Apply the shifting rules:
$H_i(N) \cdot (a^\dagger)^n = (a^\dagger)^n \cdot H_i(N-n)$ and $a^m \cdot H_j(N) = H_j(N+m) \cdot a^m$

This transforms the equation to:

$$(a^\dagger)^n \cdot [H_i(N-n) \cdot f_X(N) - f_X(N) \cdot H_j(N+m)] \cdot a^m = (a^\dagger)^n \cdot f_Y(N) \cdot a^m$$

Equate the coefficients (with $H_i$ and $H_j$ commuting with $f_X(N)$):

$$f_X(N) \cdot [H_i(N-n) - H_j(N+m)] = f_Y(N)$$

The solution is:

$$X = (a^\dagger)^n \cdot \frac{f_Y(N)}{H_i(N-n) - H_j(N+m)} \cdot a^m$$

The generalization to multiple modes follows the same pattern.
For each mode, apply the appropriate shifts to the Hamiltonian based on the creation and annihilation operators in the perturbation term.
The solution maintains the operator structure of the original perturbation.

In Pymablock, the `pymablock.second_quantization.solve_sylvester_bosonic` function implements this approach.

### Filtering terms of number ordered forms

When working with second quantized operators in perturbation theory, the goal is often often to eliminate specific terms from the operators.
Pymablock provides a format for specifying which terms to keep or eliminate based on the powers of creation and annihilation operators.

#### Matrix-Based Expression Format

Pymablock uses a matrix-based specification where SymPy expressions define elimination rules for different elements of the operator matrix:

- An expression matrix `M` has the same shape as the operator being filtered
- Each element `M[i, j]` defines the elimination rules for element `(i, j)` of the operator: every term of the number-ordered form of that element is eliminated from the corresponding element of the operator.

#### Example

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

In Pymablock, the `pymablock.second_quantization.apply_mask_to_operator` function applies these rules to the operator matrix to filter out unwanted terms.

#### Application in Block Diagonalization

When using `block_diagonalize` with second quantized operators, the `fully_diagonalize` parameter accepts this matrix-based format to control term elimination:

```python
H_tilde, *_ = block_diagonalize(
    sympy.Matrix([[H_0 + H_p]]),
    fully_diagonalize=elimination_rules,
    symbols=[g]
)
```

This approach provides fine-grained control over which quantum terms to include in the effective Hamiltonian, making it possible to implement physical approximations like the rotating wave approximation, number conservation constraints, or selective truncation of higher-order terms.
