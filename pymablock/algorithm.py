"""Parse algorithm definitions."""

import ast
import dataclasses
from collections import Counter, defaultdict
from enum import Enum
from functools import cache
from itertools import chain
from pathlib import Path


@dataclasses.dataclass
class _Series:
    """Series properties."""

    name: str = None
    start: str = None
    uses: list = dataclasses.field(default_factory=list)
    definition: list[ast.stmt] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _Product:
    """Product properties."""

    left: str = None
    right: str = None
    hermitian: list = False

    @property
    def name(self):
        return f"{self.left} @ {self.right}"


class _EvalTransformer(ast.NodeTransformer):
    def __init__(self, to_delete):
        self.to_delete = to_delete

    def visit_With(self, node):
        implicit_select = ast.parse(
            "which = linear_operator_series if use_implicit and index[0] == index[1] == 1 else series"
        ).body
        return_zero = ast.Return(value=ast.Name(id="zero", ctx=ast.Load()))

        module = ast.Module(
            body=[
                ast.FunctionDef(
                    name="series_eval",
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        vararg=ast.arg(arg="index"),
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=[
                        *implicit_select,
                        *(
                            line
                            for expr in node.body
                            for line in self._visit_Line(expr)
                            if line is not None
                        ),
                        return_zero,
                    ],
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        return module

    def _visit_Line(self, node):
        if isinstance(node, ast.Assign):
            # Delete start = ... statements
            return [None]
        if isinstance(node, ast.If):
            eval_type = _EvalType.from_condition(node.test.id)
            if eval_type is None:
                return node
            node.test = eval_type.test
            node.body = self._visit_Eval(node.body[0].value, eval_type)
            return [node]

        return self._visit_Eval(node, _EvalType.default)

    def _visit_Eval(self, node, eval_type):
        diagonal = eval_type == _EvalType.diagonal
        eval_transformers = [
            _SumTransformer(),
            _DivideTransformer(),
            _LiteralTransformer(diagonal=diagonal),
            _FunctionTransformer(),
        ]
        for transformer in eval_transformers:
            node = transformer.visit(node)
        if isinstance(node, ast.Expr):
            # TODO: Why do we get an ast.Expr here?
            node = node.value
        return [
            ast.Assign(
                targets=[ast.Name(id="result", ctx=ast.Store())],
                value=node,
            ),
            *(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Name(id="del_", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=term),
                            _LiteralTransformer._index(adjoint),
                        ],
                        keywords=[],
                    )
                )
                for term, adjoint, _eval_type in self.to_delete
                if _eval_type == eval_type
            ),
            ast.Return(value=ast.Name(id="result", ctx=ast.Load())),
        ]


class _HermitianTransformer(ast.NodeTransformer):
    """Transform hermitian attribute into if statement."""

    def __init__(self, term):
        self.term = term

    def visit_Expr(self, node):
        if not isinstance(node.value, ast.Name):
            return self.generic_visit(node)

        term = ast.Attribute(
            value=ast.Constant(value=self.term), attr="adj", ctx=ast.Load()
        )

        match node.value.id:
            case "hermitian":
                pass
            case "antihermitian":
                term = ast.UnaryOp(op=ast.USub(), operand=term)
            case _:
                return self.generic_visit(node)

        return ast.If(
            test=ast.Name(id="upper", ctx=ast.Load()),
            body=[ast.Expr(value=term)],
            orelse=[],
        )


class _UseCounter(ast.NodeVisitor):
    """Count uses of terms in an expression."""

    def __init__(self):
        self.uses = []  # List of (term, adjoint?)

    def visit_Attribute(self, node):
        # We assume the attribute is `.adj`.
        self.uses.append((node.value.value, True))

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return
        self.uses.append((node.value, False))


class _LiteralTransformer(ast.NodeTransformer):
    """Transform literals to `series[term][index]`."""

    def __init__(self, diagonal: bool):
        self.diagonal = diagonal

    def visit_Attribute(self, node):
        # We assume the attribute is `.adj`.
        return self._subscript(node.value, dagger=True)

    def visit_Constant(self, node):
        if not isinstance(node.value, str):
            return node
        return self._subscript(node, dagger=False)

    def _subscript(self, node: ast.Constant, dagger: bool):
        # Build `series[term][index] as AST
        result = ast.Subscript(
            value=ast.Subscript(
                value=ast.Name(id="which", ctx=ast.Load()),
                slice=ast.Constant(value=node.value),
                ctx=ast.Load(),
            ),
            slice=ast.Index(
                value=_LiteralTransformer._index(dagger and (not self.diagonal))
            ),
            ctx=ast.Load(),
        )
        if dagger:
            result = ast.Call(
                func=ast.Name(id="Dagger", ctx=ast.Load()),
                args=[result],
                keywords=[],
            )
        return result

    @staticmethod
    def _index(adjoint):
        if not adjoint:
            return ast.Name(id="index", ctx=ast.Load())
        return ast.parse("(index[1], index[0], *index[2:])").body[0].value


class _SumTransformer(ast.NodeTransformer):
    """Transform additive operations to zero_sum."""

    @staticmethod
    def _is_zero_sum(node):
        return isinstance(node, ast.Call) and node.func.id == "_zero_sum"

    @staticmethod
    def _zero_sum(args):
        return ast.Call(
            func=ast.Name(id="zero_sum", ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    @staticmethod
    def _negate(node):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return node.operand
        return ast.UnaryOp(
            op=ast.USub(),
            operand=node,
        )

    def visit_BinOp(self, node: ast.BinOp):
        if not (isinstance(node.op, ast.Add) or isinstance(node.op, ast.Sub)):
            return self.generic_visit(node)

        left = self.visit(node.left)
        right = self.visit(node.right)

        left_args = left.args if self._is_zero_sum(left) else [left]
        right_args = right.args if self._is_zero_sum(right) else [right]

        if isinstance(node.op, ast.Sub):
            right_args = [self._negate(arg) for arg in right_args]

        return self._zero_sum(left_args + right_args)


class _DivideTransformer(ast.NodeTransformer):
    """Replace division with safe_divide."""

    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, ast.Div):
            return self.generic_visit(node)

        return ast.Call(
            func=ast.Name(id="safe_divide", ctx=ast.Load()),
            args=[node.left, node.right],
            keywords=[],
        )


class _FunctionTransformer(ast.NodeTransformer):
    """Do not call certain functions with zero as argument."""

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)

        if node.func.id not in ["solve_sylvester"]:
            return self.generic_visit(node)

        return ast.IfExp(
            test=ast.Compare(
                left=ast.NamedExpr(
                    target=ast.Name(id="_var", ctx=ast.Store()),
                    value=node.args[0],
                ),
                ops=[ast.IsNot()],
                comparators=[ast.Name(id="zero", ctx=ast.Load())],
            ),
            body=ast.Call(
                func=node.func,
                args=[ast.Name(id="_var", ctx=ast.Load()), *node.args[1:]],
                keywords=node.keywords,
            ),
            orelse=ast.Name(id="zero", ctx=ast.Load()),
        )


def parse_algorithms(data: ast.Module):
    """Parse algorithm definitions."""
    _algorithms = {}
    for node in data.body:
        if isinstance(node, ast.FunctionDef):
            _algorithms[node.name] = parse_algorithm(node)
    return _algorithms


def parse_algorithm(definition: ast.FunctionDef):
    """Parse algorithm definition."""
    series, products, outputs = preprocess_algorithm(definition)
    to_delete = find_delete_candidates(series, products)

    for term in series:
        term.definition = _EvalTransformer(to_delete[term.name]).visit(term.definition)

    return series, products, outputs


def parse_return(node: ast.Return):
    """Parse return statement."""
    if isinstance(node, ast.Return):
        if isinstance(node.value, ast.Constant):
            return [node.value.value]
        if isinstance(node.value, ast.Tuple):
            return [element.value for element in node.value.elts]
    return []


def preprocess_algorithm(definition: ast.FunctionDef):
    """Parse algorithm series and products definition."""
    series = []
    products = []
    output = []
    for node in definition.body:
        if isinstance(node, ast.With):
            if "@" in node.items[0].context_expr.value:
                products.append(analyze_product(node))
            else:
                series.append(preprocess_series(node))
        if isinstance(node, ast.Return):
            output = parse_return(node)

    return series, products, output


def find_delete_candidates(series: list[_Series], products):
    """Determine the intermediate terms to delete."""
    terms_in_products = set(
        chain.from_iterable((product.left, product.right) for product in products)
    )
    # We should never delete terms from the original Hamiltonian.
    original_terms = {"H"}
    delete_blacklist = terms_in_products | original_terms
    uses = []
    source_map = {}
    for origin in series:
        remaining_indices = {(0, 0), (0, 1), (1, 0), (1, 1)}
        last_eval_type = None

        for term, adjoint, eval_type in origin.uses:
            if eval_type != last_eval_type:
                # Update indices used for this eval_type.
                # We assume the eval_types are ordered.
                indices = set(
                    index for index in remaining_indices if eval_type.matches(index)
                )
                remaining_indices -= indices
                last_eval_type = eval_type

            if term in delete_blacklist:
                continue

            for index in indices:
                if adjoint:
                    index = (index[1], index[0])

                uses.append((term, index))
                # This gets overwritten if the term is used multiple times.
                # This is fine since we only care about terms that are used once.
                source_map[(term, index)] = (origin.name, adjoint, eval_type)

    delete_items = [item for item, count in Counter(uses).items() if count == 1]

    result = defaultdict(set)
    for term, index in delete_items:
        origin, adjoint, eval_type = source_map[(term, index)]
        result[origin].add((term, adjoint, eval_type))
    return result


def analyze_product(definition: ast.With):
    """Analyze product properties."""
    product = _Product()
    name = definition.items[0].context_expr.value
    product.left, product.right = name.split(" @ ", maxsplit=1)
    for node in definition.body:
        if not isinstance(node, ast.Expr):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id == "hermitian":
            product.hermitian = True
    return product


class _EvalType(Enum):
    def __init__(self, test):
        self.test = ast.parse(test).body[0].value if test else None

    @staticmethod
    def from_condition(value):
        try:
            return _EvalType[value]
        except KeyError:
            return None

    def matches(self, index):
        if self.test is None:
            return True
        return eval(
            compile(ast.Expression(body=self.test), "<string>", mode="eval"),
            {},
            {"index": index},
        )

    default = (None,)
    diagonal = ("index[0] == index[1]",)
    offdiagonal = ("index[0] != index[1]",)
    upper = ("index[0] > index[1]",)


def preprocess_series(definition: ast.With):
    """Determine the properties of a series."""
    series = _Series()
    series.name = definition.items[0].context_expr.value
    series.definition = _HermitianTransformer(series.name).visit(definition)

    for node in definition.body:
        if isinstance(node, ast.Assign):
            if node.targets[0].id == "start":
                series.start = parse_start(node.value.value)
            continue

        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Name):
                continue
            eval_type = _EvalType.default
            expression = node.value
        elif isinstance(node, ast.If):
            eval_type = _EvalType.from_condition(node.test.id)
            if eval_type is None:
                continue
            expression = node.body[0].value
        else:
            continue

        (counter := _UseCounter()).visit(expression)
        series.uses += [(*use, eval_type) for use in counter.uses]

    return series


def parse_start(value: str | int):
    """Parse start value."""
    match value:
        case "H_0":
            return "H_0_data"
        case 0:
            return "zero_data"
        case 1:
            return "identity_data"


@cache
def global_scope():
    """Build the global scope for algorithm execution."""
    scope = {}
    exec("from pymablock.algorithms import *", {}, scope)
    return scope


algorithms = parse_algorithms(
    ast.parse((Path(__file__).parent / "algorithms.py").read_text())
)
