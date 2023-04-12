---
file_format: mystnb
kernelspec:
  name: python3
---
# Jaynes-Cummings model

```{code-cell}
from operator import mul

import sympy
from sympy.physics.quantum.boson import BosonOp, BosonFockKet
from sympy.physics.quantum import qapply, Dagger

from lowdin.block_diagonalization import to_BlockSeries, expanded
from lowdin.series import zero
```

```{code-cell}
# resonator frequency, qubit frequency, Rabi coupling
wr, wq, g = sympy.symbols(r"\omega_r, \omega_q, g", real=True)

# resonator photon annihilation operator
a = BosonOp("a")
```

```{code-cell}
H_0_AA = wr * Dagger(a) * a + sympy.Rational(1 / 2) * wq
H_0_BB = wr * Dagger(a) * a - sympy.Rational(1 / 2) * wq
H_p_AB = {(1,): g * Dagger(a)}
H_p_BA = {(1,): g * a}
H_p_AA = {}
H_p_BB = {}

H = to_BlockSeries(H_0_AA, H_0_BB, H_p_AA, H_p_BB, H_p_AB)
```

```{code-cell}
n = sympy.symbols("n", integer=True, positive=True)
basis_ket = BosonFockKet(n)


def expectation_value(v, operator):
    return qapply(Dagger(v) * operator * v).doit().simplify()


def norm(v):
    return sympy.sqrt(expectation_value(v, 1).factor()).doit().simplify()

def solve_sylvester(rhs):
    """
    Solves Sylvester's Equation
    rhs : zero or sympy expression

    Returns:
    V : zero or sympy expression for off-diagonal block of unitary transformation
    """
    if zero == rhs:
        return rhs

    E_i = expectation_value(basis_ket, H_0_AA)
    V = []
    for term in rhs.expand().as_ordered_terms():
        term_on_basis = qapply(term * basis_ket).doit()
        if not isinstance(norm(term_on_basis), sympy.core.numbers.Zero):
            normalized_ket = term_on_basis.as_ordered_factors()[-1]
            E_j = expectation_value(normalized_ket, H_0_BB)
            V.append(term / (E_j - E_i))
    return sum(V)
```

```{code-cell}
H_tilde, U, U_adjoint = expanded(H, solve_sylvester=solve_sylvester, op=mul)
```

```{code-cell}
H_tilde.evaluated[0, 0, 4].expand().simplify()
```
