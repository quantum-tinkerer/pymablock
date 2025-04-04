# %%
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum import Dagger
import sympy
from sympy.physics.quantum.operatorordering import normal_ordered_form

from pymablock.second_quantization import NumberOperator
# %%
def multiply_b(expr, boson):
    # Multiply the expression by b from the left
    # expr is number ordered
    n = NumberOperator(boson)

    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        if not term.has(Dagger(boson)):  # only annihilations in term
            number_ordered_terms.append(term * boson)
        else:
            # Commute n and boson
            term = term.subs(n, n - 1) * n
            # Find Dagger(boson) in term
            daggered_boson = next(
                factor
                for factor in term.as_ordered_factors()
                if factor.has(Dagger(boson))
            )
            term = term.subs(
                daggered_boson, Dagger(boson) ** (daggered_boson.as_base_exp()[1] - 1)
            )
            number_ordered_terms.append(term)
    return sympy.Add(*number_ordered_terms)


def multiply_daggered_b(expr, daggered_boson):
    # Multiply the expression by b from the left
    # expr is number ordered
    boson = Dagger(daggered_boson)
    n = NumberOperator(boson)
    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        try:
            boson_factor = next(
                (
                    factor
                    for factor in term.as_ordered_factors()
                    if factor.as_base_exp()[0] == boson
                )
            )
            term = term.subs(boson_factor, boson ** (boson_factor.as_base_exp()[1] - 1))
            number_ordered_terms.append(multiply_fn(term, n + 1))
        except StopIteration:
            # Commute n and daggered boson
            number_ordered_terms.append(daggered_boson * term.subs(n, n + 1))
    return sympy.Add(*number_ordered_terms)


def multiply_fn(expr, nexpr):
    # expr and nexpr are number ordered
    # nexpr only has number operators, but of different boson operators
    # First find all common boson operators in expr  of which nexpr has number operators
    number_ordered_terms = []
    for term in expr.as_ordered_terms():
        # Find common bosons
        boson_powers = [
            base_exp
            for factor in term.as_ordered_factors()
            if isinstance((base_exp := factor.as_base_exp())[0], BosonOp) and base_exp[0].is_annihilation
        ]
        if not boson_powers:
            number_ordered_terms.append(term * nexpr)
            continue

        fn = nexpr
        for boson, power in boson_powers:
            fn = fn.subs(NumberOperator(boson), NumberOperator(boson) + power)
        number_ordered_terms.append(fn * term)
    return sympy.Add(*number_ordered_terms)


# %%
def order_expression(expr):
    result = sympy.S.Zero
    for term in expr.as_ordered_terms():
        composed_term = sympy.S.One
        for factor in term.as_ordered_factors():
            base, power = factor.as_base_exp()
            if isinstance(base, BosonOp) and base.is_annihilation:
                if power < 0:
                    raise ValueError(
                        f"Cannot have negative power of boson operator: {base}"
                    )
                for _ in range(power):
                    composed_term = multiply_b(composed_term, base)
                continue
            elif isinstance(base, BosonOp):
                if power < 0:
                    raise ValueError(
                        f"Cannot have negative power of boson operator: {base}"
                    )
                for _ in range(power):
                    composed_term = multiply_daggered_b(composed_term, base)
                continue
            elif not base.atoms(BosonOp):  # base is a function of number operators
                composed_term = multiply_fn(composed_term, base**power)
                continue

            # Composite number-ordered expression
            base = order_expression(base)
            if power < 0 and base.atoms(BosonOp):
                raise ValueError(
                    f"Cannot have negative power of boson operator: {base}"
                )
            for _ in range(power):
                composed_term = sympy.Add(*(order_expression(composed_term * base_term) for base_term in base.as_ordered_terms()))
        result += composed_term
    return result

# %%
# n = NumberOperator(b)
# # %%
# a = n + n**2 * b + Dagger(b) * n + Dagger(b) ** 2 * n + n * b**2
# a

# TEST multiply_b

# result1 = normal_ordered_form(((a * b)).doit().expand())
# result2 = normal_ordered_form(multiply_b(a, b).doit().expand())

# display(result1)
# display(result2)
# # %%
# # TEST multiply_b
# a = NumberOperator(b)
# result1 = normal_ordered_form(((a * Dagger(b))).doit().expand())
# result2 = normal_ordered_form(multiply_daggered_b(a, Dagger(b)).doit().expand())

# display(result1)
# display(result2)
# # %%
# # TEST multiply_fn
# b = BosonOp("b")
# c = BosonOp("c")
# Nb = NumberOperator(b)
# Nc = NumberOperator(c)
# fn = (Nb**2 + 1) * Nb * (Nc + 1)

# expr = b**2 * c
# result1 = normal_ordered_form((expr * fn).doit().expand(), independent=True)
# result2 = normal_ordered_form(multiply_fn(expr, fn).doit().expand(), independent=True)
# display(result1)
# display(result2)
# %%
b = BosonOp("b")
c = BosonOp("c")
Nb = NumberOperator(b)
Nc = NumberOperator(c)
expr = b**2 * Dagger(b) * (c + 1)**2 + (Nc + 1)** (1) * (b + 1) * (c + Nb * Dagger(c))**2
display(expr)

result1 = normal_ordered_form(expr.doit().expand(), independent=True)
print('correct')
display(result1)

result2 = normal_ordered_form(order_expression(expr).doit().expand(), independent=True)
print('goal')
display(result2)
assert (result1 - result2).expand() == 0

result = order_expression(expr)
result
