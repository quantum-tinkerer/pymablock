import numpy as np
from collections import defaultdict

import sympy
import kwant


# Code below is to workaround "flood-fill" algorithm that does not
# fill systems with missing hoppings.

def discretize_with_hoppings(hamiltonian, coords=None, *, grid_spacing=1,
                             locals=None):
    """Discretize system and add zero-magnitude hoppings where required.

    This is modification of the "kwant.continuum.discretize" function
    that adds zero-magnitude hoppings in place of missing ones.

    Please check "kwant.continuum.discretize" documentation for details.
    """
    template = kwant.continuum.discretize(hamiltonian, coords,
                                          grid_spacing=grid_spacing,
                                          locals=locals)

    syst = kwant.Builder(template.symmetry)
    lat = template.lattice

    syst[next(iter(template.sites()))] = np.zeros((lat.norbs, lat.norbs))
    syst[lat.neighbors()] = np.zeros((lat.norbs, lat.norbs))

    syst.update(template)
    return syst


# Function defined in this section come from "kwant.continuum" module
# of Kwant and are currently a part of a non-public API.
# To avoid breakage with future releases, they are defined here.

def make_commutative(expr, *symbols):
    """Make sure that specified symbols are defined as commutative.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
    symbols: sequace of symbols
        Set of symbols that are requiered to be commutative. It doesn't matter
        of symbol is provided as commutative or not.

    Returns
    -------
    input expression with all specified symbols changed to commutative.
    """
    names = [s.name if not isinstance(s, str) else s for s in symbols]
    symbols = [sympy.Symbol(name, commutative=False) for name in names]
    expr = expr.subs({s: sympy.Symbol(s.name) for s in symbols})
    return expr


def monomials(expr, gens=None):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
        Sympy expression to be parsed into monomials.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of monomials. If unset it will default to all
        symbols used in ``expr``.

    Returns
    -------
    dictionary (generator: monomial)

    Example
    -------
        >>> expr = kwant.continuum.sympify("A * (x**2 + y) + B * x + C")
        >>> monomials(expr, gens=('x', 'y'))
        {1: C, x: B, x**2: A, y: A}
    """
    if gens is None:
        gens = expr.atoms(sympy.Symbol)
    else:
        gens = [kwant.continuum.sympify(g) for g in gens]

    if not isinstance(expr, sympy.MatrixBase):
        return _expression_monomials(expr, gens)
    else:
        output = defaultdict(lambda: sympy.zeros(*expr.shape))
        for (i, j), e in np.ndenumerate(expr):
            mons = _expression_monomials(e, gens)
            for key, val in mons.items():
                output[key][i, j] += val
        return dict(output)


def _expression_monomials(expr, gens):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr
        Sympy expr to be parsed.
    gens: sequence of sympy.Symbol
        Generators of monomials.

    Returns
    -------
    dictionary (generator: monomial)
    """
    expr = sympy.expand(expr)
    output = defaultdict(lambda: sympy.Integer(0))
    for summand in expr.as_ordered_terms():
        key = []
        val = []
        for factor in summand.as_ordered_factors():
            symbol, exponent = factor.as_base_exp()
            if symbol in gens:
                key.append(factor)
            else:
                val.append(factor)
        output[sympy.Mul(*key)] += sympy.Mul(*val)

    return dict(output)


def exact_greens_function(ham):
    """Takes a Hamiltonian and returns the Green's function operator."""
    eigs, evecs = np.linalg.eigh(ham)
    (dim,) = eigs.shape
    def green(vec, e, eta=1e-2j):
        """Takes a vector `vec` of shape (M,N), with `M` vectors of length `N`,
        the same as the Hamiltonian. Returns the Green's function exact expansion
        of the vectors with the same shape as `vec`."""
        # normalize the shapes of `e` and `vec`
        e = np.atleast_1d(e).flatten()
        (num_e,) = e.shape
        vec = np.atleast_2d(vec)
        num_vectors, vec_dim = vec.shape
        assert vec_dim == dim
        assert num_vectors == num_e

        coefs = vec @ evecs.conj()
        e_diff = e[:,None] - eigs[None,:]
        coefs = coefs / (e_diff + eta)
        return coefs @ evecs.T
    return green
