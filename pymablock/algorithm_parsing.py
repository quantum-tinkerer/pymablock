"""Parse algorithms from the custom syntax to executable AST."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections import Counter, defaultdict
from enum import Enum
from itertools import chain


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
    def name(self) -> str:
        """Name that represents the product."""
        return f"{self.left} @ {self.right}"


class _EvalTransformer(ast.NodeTransformer):
    """Transforms a `with` statement to a callable eval understood by `BlockSeries`."""

    def __init__(self, to_delete):
        self.to_delete = to_delete

    def visit_With(self, node: ast.With) -> ast.Module:
        """Build a function definition from a `with` statement."""
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

    def _visit_Line(self, node: ast.AST) -> list[ast.AST]:
        """Transform each line of the function body.

        Assign statements are removed.
        If statements are transformed to a valid index test.
        Expressions are transformed using `_visit_Eval`.
        """
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

    def _visit_Eval(self, node: ast.Eval, eval_type: _EvalType) -> list[ast.AST]:
        """Transform evaluation expressions to executable AST.

        First it applies `_SumTransformer`, `_DivideTransformer`, `_LiteralTransformer` and `_FunctionTransformer`
        and stores the result.
        Then it inserts delete statements for intermediate terms.
        Finally it returns the result.
        """
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
    """Transform hermitian attributes into if statements."""

    def __init__(self, term):
        self.term = term

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """Insert a conditional evaluation for hermitian and antihermitian attributes.

        It adds an if statement with `upper` that matches indices in the upper triangle.
        The evaluation result is either the conjugate adjoint of itself in case of `hermitian`
        or the negation of that in case of `antihermitian`.
        """
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
    """Count uses of terms in an expression.

    The result is later used to determine which terms are accessed exactly once,
    which can be deleted from the series after accessing them.
    """

    def __init__(self):
        self.uses = []  # List of (term, adjoint)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Count an adjoint access."""
        # We assume the attribute is `.adj`.
        self.uses.append((node.value.value, True))

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Count a regular access."""
        if not isinstance(node.value, str):
            return
        self.uses.append((node.value, False))


class _LiteralTransformer(ast.NodeTransformer):
    """Transform string literals to `series[term][index]`."""

    def __init__(self, diagonal: bool):
        self.diagonal = diagonal

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Transform adjoint terms."""
        # We assume the attribute is `.adj`.
        return self._to_series(node.value, adjoint=True)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Transform regular terms."""
        if not isinstance(node.value, str):
            return node
        return self._to_series(node, adjoint=False)

    def _to_series(self, node: ast.Constant, adjoint: bool) -> ast.AST:
        """Build series[term][index] as AST."""
        result = ast.Subscript(
            value=ast.Subscript(
                value=ast.Name(id="which", ctx=ast.Load()),
                slice=ast.Constant(value=node.value),
                ctx=ast.Load(),
            ),
            slice=ast.Index(value=self._index(adjoint and (not self.diagonal))),
            ctx=ast.Load(),
        )
        if adjoint:
            result = ast.Call(
                func=ast.Name(id="Dagger", ctx=ast.Load()),
                args=[result],
                keywords=[],
            )
        return result

    @staticmethod
    def _index(adjoint) -> ast.AST:
        """Build the (adjoint) index as AST."""
        if not adjoint:
            return ast.Name(id="index", ctx=ast.Load())
        return ast.parse("(index[1], index[0], *index[2:])").body[0].value


class _SumTransformer(ast.NodeTransformer):
    """Transform additive operations to `_zero_sum`."""

    @staticmethod
    def _is_zero_sum(node: ast.AST) -> bool:
        """Whether a node is a call to `_zero_sum`."""
        return isinstance(node, ast.Call) and node.func.id == "_zero_sum"

    @staticmethod
    def _zero_sum(args: list[ast.AST]) -> ast.Call:
        """Build AST representation of `_zero_sum` of args."""
        return ast.Call(
            func=ast.Name(id="_zero_sum", ctx=ast.Load()),
            args=args,
            keywords=[],
        )

    @staticmethod
    def _negate(node: ast.AST) -> ast.AST:
        """Negate a node. Return the original node if already negated."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return node.operand
        return ast.UnaryOp(
            op=ast.USub(),
            operand=node,
        )

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Recursively transform subtraction and addition to a `_zero_sum` call."""
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
    """Replace division with `_safe_divide`."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Transform division to a `_safe_divide` call."""
        if not isinstance(node.op, ast.Div):
            return self.generic_visit(node)

        return ast.Call(
            func=ast.Name(id="_safe_divide", ctx=ast.Load()),
            args=[node.left, node.right],
            keywords=[],
        )


class _FunctionTransformer(ast.NodeTransformer):
    """Do not call certain functions with zero as argument."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Apply a zero check to calls of `solve_sylvester`."""
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


def _parse_algorithm(
    definition: ast.FunctionDef,
) -> tuple[list[_Series], list[_Product], list[str]]:
    """Parse full algorithm definition.

    First it reads the series, products and output properties.
    Then it determines the intermediate terms to delete.
    Finally it transforms each series definition to executable AST.

    Returns the series, product and outputs definitions.
    """
    series, products, outputs = _preprocess_algorithm(definition)
    to_delete = _find_delete_candidates(series, products, outputs)

    for term in series:
        term.definition = _EvalTransformer(to_delete[term.name]).visit(term.definition)

    return series, products, outputs


def _parse_return(node: ast.Return) -> list[str]:
    """Parse return statement to list of series names."""
    if isinstance(node, ast.Return):
        if isinstance(node.value, ast.Constant):
            return [node.value.value]
        if isinstance(node.value, ast.Tuple):
            return [element.value for element in node.value.elts]
    return []


def _preprocess_algorithm(
    definition: ast.FunctionDef,
) -> tuple[list[_Series], list[_Product], list[str]]:
    """Read and preprocess series, products and outputs definition."""
    series = []
    products = []
    outputs = []
    for node in definition.body:
        if isinstance(node, ast.With):
            if "@" in node.items[0].context_expr.value:
                products.append(_read_product(node))
            else:
                series.append(_preprocess_series(node))
        if isinstance(node, ast.Return):
            outputs = _parse_return(node)

    return series, products, outputs


def _find_delete_candidates(
    series: list[_Series], products: list[_Product], outputs: list[str]
) -> dict[str, list[tuple[str, tuple[int, int], "_EvalType"]]]:
    """Determine the intermediate terms to delete.

    All terms that are accessed exactly once are detected.

    The result is a dictionary where the values correspond to the terms to be deleted, consisting of the name, index and eval type.
    Each key marks from which series the terms are accessed.

    """
    # We should never delete terms that appear in products,
    # are part of the original Hamiltonian or are part of the output.
    terms_in_products = set(
        chain.from_iterable((product.left, product.right) for product in products)
    )
    original_terms = {"H"}
    delete_blacklist = terms_in_products | original_terms | set(outputs)

    uses = []  # List of (term, index)
    source_map = {}  # Map from (term, index) to (origin, adjoint, eval_type)

    # Collect all accessed terms and indices of the entire algorithm.
    for origin in series:
        remaining_indices = {(0, 0), (0, 1), (1, 0), (1, 1)}
        last_eval_type = None

        for term, adjoint, eval_type in origin.uses:
            if eval_type != last_eval_type:
                # Update indices used for this eval_type.
                # We assume the uses are ordered by their appearance in the series definition.
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

    # Find terms that are used exactly once.
    delete_items = [item for item, count in Counter(uses).items() if count == 1]

    # Group terms by their origin and collect the adjoint and eval_type.
    result = defaultdict(set)
    for term, index in delete_items:
        origin, adjoint, eval_type = source_map[(term, index)]
        result[origin].add((term, adjoint, eval_type))

    return result


def _read_product(definition: ast.With) -> _Product:
    """Read product properties."""
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
    """Represents the different types of evaluations."""

    def __init__(self, test: str):
        self.test = ast.parse(test).body[0].value if test else None

    @staticmethod
    def from_condition(value: str) -> _EvalType | None:
        """Get the eval type from an if statement test."""
        try:
            return _EvalType[value]
        except KeyError:
            return None

    def matches(self, index: tuple[int, int]) -> bool:
        """Whether the index matches the condition of this eval type."""
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


def _preprocess_series(definition: ast.With) -> _Series:
    """Determine the properties of a series."""
    series = _Series()
    series.name = definition.items[0].context_expr.value
    series.definition = _HermitianTransformer(series.name).visit(definition)

    for node in definition.body:
        # Read and remove start = ... statements.
        if isinstance(node, ast.Assign):
            if node.targets[0].id == "start":
                series.start = _parse_start(node.value.value)
            continue

        # Extract the expression and eval type.
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

        # Count uses of terms in the expression to later determine candidates for deletion.
        (counter := _UseCounter()).visit(expression)
        series.uses += [(*use, eval_type) for use in counter.uses]

    return series


def _parse_start(value: str | int) -> str:
    """Parse start value."""
    match value:
        case "H_0":
            return "H_0_data"
        case 0:
            return "zero_data"
        case 1:
            return "identity_data"


def algorithm(func: callable) -> tuple[list[_Series], list[_Product], list[str]]:
    """Turn a function into an algorithm.

    Each algorithm is represented by a function definition which needs to be decorated with this function.
    The function body contains multiple `with` statements that define the series and products of that algorithm.
    Throughout the definition the series and products are represented by their name using string literals.

    A series definition allows the following statements:
    - `start = ...` to define the start value of the series. Allowed values are "H_0", 0, 1.
    - `hermitian` or `antihermitian` to optionally mark the offdiagonal blocks of the series as hermitian or antihermitian.
    - An expression that defines how to evaluate the series. The expression can contain the following:
        - String literals to represent series.
        - Attribute `.adj` access to represent the conjugate adjoint of a series.
        - Integer literals.
        - Unary and binary operations.
        - Function calls.
    - `if <condition>:` to differentiate evaluation based on the requested index. Allowed conditions are:
        - `diagonal`: indices on the main diagonal.
        - `offdiagonal`: indices *not* on the main diagonal.
        - `upper`: indices in the upper triangle.

    If a name contains an "@" symbol, it is considered a product of the left and right series.
    A product definition allows the following statements:
    - `hermitian` to mark the product as hermitian.
    - `pass` if the product requires no statements.

    The final return statement in the function body defines a tuple of series that are part of the output of the algorithm.

    Arguments:
    ---------
    func :
        The module containing the algorithm definitions.

    Returns:
    -------
    algorithm :
        A tuple containing the series, products, and outputs of the algorithm.

    """
    source = ast.parse(inspect.getsource(func))
    function_def = source.body[0]
    return _parse_algorithm(function_def)
