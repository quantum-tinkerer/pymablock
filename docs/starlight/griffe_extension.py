"""Retain the explicitly documented Sphinx API beyond module __all__ lists."""

from griffe import Extension


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
