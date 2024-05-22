"""Parse algorithm definitions."""

import ast
import dataclasses


@dataclasses.dataclass
class Series:
    """Series definition."""

    name: str = None
    start: str | int = None
    hermitian: bool = False
    antihermitian: bool = False
    default_eval: str = None
    diagonal_eval: str = None
    offdiagonal_eval: str = None


def parse_algorithms(data: ast.Module):
    """Parse algorithm definitions."""
    algorithms = {}
    for node in data.body:
        if isinstance(node, ast.FunctionDef):
            algorithms[node.name] = list(parse_algorithm(node))
    return algorithms


def parse_algorithm(definition: ast.FunctionDef):
    """Parse an algorithm definition."""
    for node in definition.body:
        if isinstance(node, ast.With):
            yield parse_series(node)

    # TODO: Parse return statement


def parse_series(definition: ast.With):
    """Parse a series."""
    series = Series()

    if len(definition.items) != 1 or not isinstance(
        name := definition.items[0].context_expr, ast.Constant
    ):
        raise ValueError("Series name must be a literal.")
    series.name = name.value

    for node in definition.body:
        if isinstance(node, ast.Assign):
            parse_assign(node, series)
        elif isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Name):
                parse_property(node.value, series)
            else:
                series.default_eval = parse_eval(node.value)
        if isinstance(node, ast.If):
            parse_condition(node, series)

    return series


def parse_property(name: ast.Name, series: Series):
    """Parse a property."""
    match name.id:
        case "hermitian":
            series.hermitian = True
        case "antihermitian":
            series.antihermitian = True
        case _:
            raise ValueError(f"Unknown series property: {name.id}")


def parse_assign(assign: ast.Assign, series: Series):
    """Parse an assignment."""
    if len(assign.targets) != 1:
        raise ValueError("Cannot assign multiple targets.")
    target = assign.targets[0]
    if not isinstance(target, ast.Name):
        raise ValueError("Assignment target must be a name.")
    if not isinstance(assign.value, ast.Constant):
        raise ValueError("Assignment value must be a literal.")
    match target.id:
        case "start":
            series.start = assign.value.value
        case _:
            raise ValueError(f"Unknown series property: {target.id}")


def parse_eval(expr: ast.Expr | ast.expr):
    """Parse an evaluation expression."""
    adjoint = False
    if isinstance(expr, ast.Attribute):
        if expr.attr != "adj":
            raise ValueError(f"Unsupported attribute: {expr.attr}")
        adjoint = True
        expr = expr.value
    if isinstance(expr, ast.Constant):
        if isinstance(expr.value, str):
            index = "(index[1], index[0], *index[:2])" if adjoint else "index"
            return f'which["{expr.value}"][{index}]'
        if isinstance(expr.value, int):
            return str(expr.value)
        raise ValueError(f"Unsupported constant: {expr.value}")
    if isinstance(expr, ast.BinOp):
        if isinstance(expr.op, ast.Add) or isinstance(expr.op, ast.Sub):
            return f"""zero_sum({', '.join(
                f"{term}" if factor == 1 else f"(-{term})"
                for factor, term in collect_terms(expr))})"""
        if isinstance(expr.op, ast.Div):
            return f"safe_divide({parse_eval(expr.left)}, {parse_eval(expr.right)})"
        raise ValueError(f"Unsupported operator: {expr.op}")
    if isinstance(expr, ast.UnaryOp):
        if not (isinstance(expr.op, ast.USub) or expr.op == ast.USub):
            raise ValueError(f"Unsupported unary operator: {expr.op}")
        return f"-({parse_eval(expr.operand)})"
    if isinstance(expr, ast.Call):
        if not isinstance(expr.func, ast.Name):
            raise ValueError(f"Unsupported function: {expr.func}")
        if len(expr.args) != 1:
            raise ValueError(f"Unsupported number of arguments: {expr.args}")
        match expr.func.id:
            case "solve_sylvester":
                return f"(solve_sylvester(Y) if (Y := {parse_eval(expr.args[0])}) is not zero else zero)"
            case _:
                raise ValueError(f"Unsupported function: {expr.func}")
    raise ValueError(f"Unsupported part of evaluation function: {expr}")


def collect_terms(expr: ast.expr | ast.BinOp):
    """Collect all terms of neighbouring additive operators."""
    if not isinstance(expr, ast.BinOp):
        yield 1, parse_eval(expr)
    elif isinstance(expr.op, ast.Add):
        yield from collect_terms(expr.left)
        yield from collect_terms(expr.right)
    elif isinstance(expr.op, ast.Sub):
        yield from collect_terms(expr.left)
        yield from ((-1 * factor, term) for factor, term in collect_terms(expr.right))
    else:
        yield 1, parse_eval(expr)


def parse_condition(expr: ast.If, series: Series):
    """Parse conditional eval."""
    if not isinstance(expr.test, ast.Name):
        raise ValueError(f"Unsupported condition: {expr.test}")
    if len(expr.body) != 1:
        raise ValueError("Cannot define multiple evaluation functions.")
    if not isinstance(body := expr.body[0], ast.Expr):
        raise ValueError("Evaluation function should be an expression.")
    match expr.test.id:
        case "diagonal":
            series.diagonal_eval = parse_eval(body.value)
        case "offdiagonal":
            series.offdiagonal_eval = parse_eval(body.value)
        case "_":
            raise ValueError(f"Unsupported condition: {expr.test.id}")
