"""Retain the explicitly documented Sphinx API beyond module __all__ lists."""

from pathlib import Path

from griffe import Attribute, Docstring, Extension, load_extensions


class ExplicitMembers(Extension):
    """Expose configured documentation targets without changing Python exports."""

    members = (
        "pymablock.number_ordered_form.LadderOp",
        "pymablock.block_diagonalization.solve_sylvester_diagonal",
        "pymablock.block_diagonalization.solve_sylvester_direct",
        "pymablock.block_diagonalization.solve_sylvester_KPM",
    )

    def on_package(self, *, pkg, **_kwargs):
        """Include the members explicitly documented by the original API reference."""
        for path in self.members:
            module_path, name = path.rsplit(".", 1)
            module = (
                pkg
                if module_path == pkg.path
                else pkg[module_path.removeprefix(pkg.path + ".")]
            )
            module.members[name]
            if module.exports is not None:
                module.exports = set(module.exports) | {name}

        # Sphinx autodata obtains these instance docstrings from their classes.
        # Griffe keeps the class definitions even though Python deletes them.
        series = pkg["series"]
        for name in ("one", "zero"):
            series[name].docstring = Docstring(
                series[name.title()].docstring.value, parent=series[name], parser="numpy"
            )

        # SymPy's metaclass creates these public properties dynamically. Enrich
        # only the explicitly documented class, not its entire inherited API.
        from pymablock.number_ordered_form import NumberOrderedForm

        cls = pkg["number_ordered_form.NumberOrderedForm"]
        for name, descriptor in vars(NumberOrderedForm).items():
            if (
                name.startswith("is_")
                and isinstance(descriptor, property)
                and name not in cls.members
            ):
                member = Attribute(name, annotation="bool | None")
                member.labels.add("property")
                member.docstring = Docstring(
                    descriptor.__doc__
                    or f"SymPy assumption query ``{name}``. Returns ``True`` or ``False`` when determined, and ``None`` when unknown.",
                    parent=member,
                    parser="numpy",
                )
                cls.set_member(name, member)

        load_extensions(
            {
                str(
                    Path(__file__).parent
                    / "node_modules/astro-myst-notebooks/dist/griffe_myst.py"
                ): {
                    "exclude": ["pymablock.tests", "pymablock.tests.*"],
                    "class_only": [
                        "pymablock.number_ordered_form.LadderOp",
                        "pymablock.number_ordered_form.NumberOrderedForm",
                    ],
                    "documents": {"../second_quantization": "/second_quantization/"},
                }
            }
        ).call("on_package", pkg=pkg)
