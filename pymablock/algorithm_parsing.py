"""Tools for compiling optimized series computations."""

from __future__ import annotations

import ast
import dataclasses
import inspect
from collections import Counter, defaultdict
from collections.abc import Callable  # noqa: TCH003 (sphinx needs unconditional import)
from enum import Enum
from functools import cache
from itertools import chain
from operator import matmul
from typing import Any

import numpy as np
from sympy.physics.quantum import Dagger

from pymablock.linalg import aslinearoperator
from pymablock.series import BlockSeries, cauchy_dot_product, one, zero

__all__ = ["series_computation", "parse_algorithm"]

result = ast.Name(id="result", ctx=ast.Load())


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

    terms: list[str] = dataclasses.field(default_factory=list)
    hermitian: list = False

    @property
    def name(self) -> str:
        """Name that represents the product."""
        return " @ ".join(self.terms)


class _EvalTransformer(ast.NodeTransformer):
    """Transforms a `with` statement to a callable eval understood by `BlockSeries`."""

    def __init__(self, to_delete):
        self.to_delete = to_delete

    def visit_With(self, node: ast.With) -> ast.Module:
        """Build a function definition from a `with` statement."""
        linear_operator_select = ast.parse(
            "which = linear_operator_series if use_linear_operator[index[:2]] else series"
        ).body
        result_is_zero = ast.parse("result = zero").body

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
                        *linear_operator_select,
                        *result_is_zero,
                        *(
                            line
                            for expr in node.body
                            for line in self._visit_Line(expr)
                            if line is not None
                        ),
                        ast.Return(value=result),
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
            # The diagonal blocks are wrapped inside `diag`
            if eval_type == _EvalType.diagonal:
                node.body[0] = ast.Expr(
                    ast.Call(
                        ast.Name(id="diag", ctx=ast.Load()), [node.body[0].value], []
                    )
                )
            nodes = [node]
            # If an offdiagonal eval is present, we need to evaluate
            # this wrapped with `offdiag` for diagonal blocks.
            if eval_type == _EvalType.offdiagonal:
                nodes.append(
                    ast.If(
                        test=ast.BoolOp(
                            op=ast.And(),
                            values=[
                                ast.parse("offdiag is not None").body[0].value,
                                _EvalType.diagonal.test,
                            ],
                        ),
                        body=[
                            ast.Expr(
                                ast.Call(
                                    ast.Name(id="offdiag", ctx=ast.Load()),
                                    [node.body[0].value],
                                    [],
                                )
                            )
                        ],
                        orelse=[],
                    )
                )
            for node in nodes:
                node.body = self._visit_Eval(node.body[0], eval_type)
            if eval_type == _EvalType.lower:
                nodes[0].body.append(ast.Return(value=result))

            return nodes

        return self._visit_Eval(node, _EvalType.default)

    def _visit_Eval(self, node: ast.Expr, eval_type: _EvalType) -> list[ast.AST]:
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
            _FunctionTransformer(),
            _LiteralTransformer(diagonal=diagonal),
        ]
        node = node.value  # Get the expression from the Expr node.
        node = ast.BinOp(left=result, op=ast.Add(), right=node)
        for transformer in eval_transformers:
            node = transformer.visit(node)
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
        ]


class _HermitianTransformer(ast.NodeTransformer):
    """Transform hermitian attributes into if statements."""

    def __init__(self, term):
        self.term = term

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """Insert a conditional evaluation for hermitian and antihermitian attributes.

        It adds an if statement with `lower` that matches indices in the lower triangle.
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
            test=ast.Name(id="lower", ctx=ast.Load()),
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

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        # Do not visit subscripts as these are already transformed.
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Transform adjoint terms."""
        # We assume the attribute is `.adj`.
        return self._to_series_index(node.value, adjoint=True)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Transform regular terms."""
        if not isinstance(node.value, str):
            return node
        return self._to_series_index(node, adjoint=False)

    @staticmethod
    def _to_series(node: ast.Constant) -> ast.AST:
        """Build series[term] as AST."""
        return ast.Subscript(
            value=ast.Name(id="which", ctx=ast.Load()),
            slice=ast.Constant(value=node.value),
            ctx=ast.Load(),
        )

    def _to_series_index(self, node: ast.Constant, adjoint: bool) -> ast.AST:
        """Build series[term][index] as AST."""
        result = ast.Subscript(
            value=self._to_series(node),
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
    """Transforms function calls.

    The internal functions `_safe_divide` and `_zero_sum` are not modified.
    Other functions are changed as follows:
    - If an argument to the function is a series, it is transformed into `series["arg"]`.
    - All other arguments are left unchanged.
    - The index is added as the last argument.
    """

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if not isinstance(node.func, ast.Name):
            return self.generic_visit(node)

        # Functions introduced internally, should not be modified.
        if node.func.id in ["_safe_divide", "_zero_sum"]:
            return self.generic_visit(node)

        return self._visit_series_argument(node)

    def _visit_series_argument(self, node: ast.Call) -> ast.AST:
        """Transform functions that have series as arguments.

        Inserts the index as the last argument, preceded by all series passed as arguments.
        The series arguments are string literals, which are transformed to `series["arg"]`.
        """
        node.args = [
            *(
                _LiteralTransformer._to_series(arg)
                if (isinstance(arg, ast.Constant) and isinstance(arg.value, str))
                else arg
                for arg in node.args
            ),
            ast.Name(id="index", ctx=ast.Load()),
        ]
        return node


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

    The result is a dictionary where the values correspond to the terms to be
    deleted, consisting of the name, index and eval type. Each key marks from
    which series the terms are accessed.

    """
    # We should never delete terms that appear in products, are part of the input, or
    # are part of the output.
    terms_in_products = set(chain.from_iterable(product.terms for product in products))
    computed = set(term.name for term in series)
    inputs = (
        set(
            needed_term
            for term in series
            for needed_term, _, _ in term.uses
            if "@" not in needed_term  # Products are defined in a different way.
        )
        - computed
    )
    delete_blacklist = terms_in_products | inputs | set(outputs)

    uses = []  # List of (term, index)
    source_map = {}  # Map from (term, index) to (origin, adjoint, eval_type)

    # Collect all accessed terms and indices of the entire algorithm.
    for origin in series:
        # These indices are valid for 2x2 matrices, but with larger sizes they
        # keep track of the diagonal/offdiagonal structure.
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
    product.terms = name.split(" @ ")
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
    lower = ("index[0] > index[1]",)


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
        case str(name):
            return name + "_data"
        case 0:
            return "zero_data"
        case 1:
            return "identity_data"


@cache
def parse_algorithm(func: Callable) -> tuple[list[_Series], list[_Product], list[str]]:
    """Turn a function into an algorithm.

    Each algorithm is represented by a function definition.

    See the `series_computation` function for a more complete format description.

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


def _zero_sum(*terms: Any) -> Any:
    """Sum that returns a singleton zero if empty and omits zero terms.

    Parameters
    ----------
    terms :
        Terms to sum over with zero as default value.

    Returns
    -------
    Sum of terms, or zero if terms is empty.

    """
    return sum((term for term in terms if term is not zero), start=zero)


def _safe_divide(numerator, denominator):
    """Divide unless it's impossible, then multiply by inverse."""
    try:
        return numerator / denominator
    except TypeError:
        return numerator * (1 / denominator)


def series_computation(
    series: dict[str, BlockSeries],
    algorithm: Callable,
    scope: dict | None = None,
    *,
    operator: Callable | None = None,
) -> tuple[dict[str, BlockSeries], dict[str, BlockSeries]]:
    """Compile a `~pymablock.series.BlockSeries` computation.

    Given several series, functions to apply to their elements, and an algorithm,
    return the output series defined by the algorithm.

    The algorithm parsing used by this function applies multiple optimizations used in
    the Pymablock main algorithm. While these could be generated by hand, the resulting
    code is complex and repetitive. The mini-language used to specify the algorithm
    allows to avoid this complexity and it is useful to improve the current algorithm as
    well as develop new ones.

    Specifically, algorithm parsing:

    - Handles initialization of series and the definition of their evaluation functions.
    - Utilizes hermiticity and antihermiticity to reduce the number of evaluations.
    - Automatically handles the implicit mode when parts of the series are provided as
      linear operators.
    - Handles deletion of intermediate series terms that are only used once to reduce
      the memory usage.

    Implementing a new algorithm is advanced usage, and familiarity with the codebase is
    highly recommended.

    Parameters
    ----------
    series :
        Dictionary with all input series, where the keys are the names of the series.
    algorithm :
        Algorithm to use for the block diagonalization.  Should be passed as a callable
        whose contents follow the algorithm mini-language, see notes below.
    scope :
        Extra variables to pass to pass to the algorithm. It is particularly relevant
        for passing custom functions or data.
    operator :
        (optional) function to use for matrix multiplication.
        Defaults to matmul.

    Returns
    -------
    series : dict[str, BlockSeries]
        A dictionary with all the series used in the computation. The keys are
        the names of the series and the values are the corresponding
        `~pymablock.series.BlockSeries`.
    linear_operator_series : dict[str, BlockSeries]
        The same series as above, but wrapped into linear operators. Only used
        in the implicit mode.

    Notes
    -----
    The ``algorithm`` callable is not evaluated directly, but rather parsed to extract
    the computation that needs to be performed. It needs to follow the
    specification below.

    .. warning::

       This domain-specific language is experimental and may change in the future.

    The function body contains multiple `with` statements that define the series and
    products of that algorithm. Throughout the definition the series and products are
    represented by their name using string literals.

    A series definition allows the following statements:

    - ``start = ...`` to define the zeroth order of the series. Allowed values are
      ``"series_name"``, ``0``, ``1``.
    - ``hermitian`` or ``antihermitian`` to optionally mark the lower triangular blocks
      of a series to be evaluated using a conjugate transpose of the upper triangular
      blocks.
    - One or more expressions that define how to evaluate the series. If there are
      multiple expressions, they are summed together. The expression can contain the
      following:

      - String literals to represent series.
      - Attribute ``.adj`` access to represent the conjugate adjoint of a series.
      - Integer literals.
      - Unary and binary operations.
      - Function calls. Using ``f("series")`` will call the function ``f`` with the
        series and the block index as arguments. Using ``f(expression)`` will call the
        function with the evaluated expression and block index as arguments.

    - ``if <condition>:`` differentiates evaluation based on the requested index.
      Allowed conditions are:

      - ``diagonal``: indices on the main diagonal.
      - ``offdiagonal``: indices *not* on the main diagonal.
      - ``lower``: indices in the lower triangle.

    If a name contains an "@" symbol, it defines a Cauchy product of the terms in it.
    For example ``"A @ B @ C"`` is a Cauchy product of the series ``A``, ``B``, and
    ``C``.

    A product definition must contain one of the two following statements:

    - ``hermitian`` to mark the product as hermitian.
    - ``pass`` otherwise.

    The final return statement in the function body defines a tuple of series that are
    part of the output of the algorithm.

    Example
    -------
    The algorithm definition may look as follows (this example does not do anything
    useful):

    .. code-block:: python

        def my_algorithm():
            with "B":
                start = 0
                hermitian
                if diagonal:
                    "A" + f("B @ C")

            with "C":
                start = "A"
                if offdiagonal:
                    "A" + "B" / 2
                "B @ C"

            with "B @ C":
                hermitian

            return "C"

    Here ``"A"`` is an input, ``"B"`` and ``"C"`` are defined in the computation, and
    the function ``f`` must be provided using the scope.

    For an extended example, see the ``main`` function in ``pymablock/algorithms.py``.

    """
    if operator is None:
        operator = matmul

    # For now we demand that all series are similar because outputs are like inputs.
    dimension_names = next(iter(series.values())).dimension_names
    if any(series.dimension_names != dimension_names for series in series.values()):
        raise ValueError("All series must have the same dimension names.")

    n_infinite = {series.n_infinite for series in series.values()}
    if len(n_infinite) > 1:
        raise ValueError("All series must have the same number of infinite indices.")
    n_infinite = next(iter(n_infinite))
    shape = next(iter(series.values())).shape

    zeroth_order = (0,) * n_infinite
    all_blocks = [(i, j) for i in range(shape[0]) for j in range(shape[1])]
    diagonal = [(i, i) for i in range(shape[0])]
    zero_data = {block + zeroth_order: zero for block in all_blocks}
    identity_data = {block + zeroth_order: one for block in diagonal}
    data = {
        "zero_data": zero_data,
        "identity_data": identity_data,
        **{
            f"{name}_0_data": {
                block + zeroth_order: series[block + zeroth_order] for block in all_blocks
            }
            for name, series in series.items()
        },
    }

    # Common series kwargs to avoid some repetition
    series_kwargs = dict(
        shape=shape,
        n_infinite=n_infinite,
        dimension_names=dimension_names,
    )

    def linear_operator_wrapped(original: BlockSeries) -> BlockSeries:
        return BlockSeries(
            eval=(lambda *index: aslinearoperator(original[index])),
            name=original.name,
            **series_kwargs,
        )

    linear_operator_series = {
        name: linear_operator_wrapped(series) for name, series in series.items()
    }

    def del_(series_name, index: int) -> None:
        series[series_name].pop(index, None)
        linear_operator_series[series_name].pop(index, None)

    eval_scope = {
        # Defined in this function
        "series": series,
        "linear_operator_series": linear_operator_series,
        "del_": del_,
        "use_linear_operator": np.zeros(shape, dtype=bool),
        "offdiag": None,
        "diag": lambda x, index: x[index] if isinstance(x, BlockSeries) else x,
        # Globals
        "zero": zero,
        "_safe_divide": _safe_divide,
        "_zero_sum": _zero_sum,
        "Dagger": Dagger,
        # User-provided, may override the above
        **(scope or {}),
    }

    terms, products, outputs = parse_algorithm(algorithm)

    for term in terms:
        # This defines `series_eval` as the eval function for this term.
        exec(compile(term.definition, filename="<string>", mode="exec"), eval_scope)

        series_data = data.get(term.start, None)

        series[term.name] = BlockSeries(
            eval=eval_scope["series_eval"],
            data=series_data,
            name=term.name,
            **series_kwargs,
        )
        linear_operator_series[term.name] = linear_operator_wrapped(series[term.name])

    for product in products:
        for which in series, linear_operator_series:
            which[product.name] = cauchy_dot_product(
                *(which[term] for term in product.terms),
                operator=operator,
                hermitian=product.hermitian,
            )

    return series, linear_operator_series
