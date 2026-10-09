"""Build the Lean proofs and render their checked correspondence ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "formalization"
REPORTS = PROJECT / ".lake" / "reports"
ELAN_URL = (
    "https://github.com/leanprover/elan/releases/download/v4.2.4/"
    "elan-x86_64-unknown-linux-gnu.tar.gz"
)
ELAN_SHA256 = "42b94d4244e8353142c456ec0e4ca6528fd898a6c604d4059f494e706e431f63"


def lake_command(install: bool = False) -> list[str]:
    """Use the official toolchain so mathlib's matching build cache is usable."""
    elan_bin = Path(os.environ.get("ELAN_HOME", Path.home() / ".elan")) / "bin"
    lake = elan_bin / "lake"
    if lake.is_file():
        return [str(lake)]
    if found := shutil.which("lake"):
        return [found]
    if not install:
        raise SystemExit("Lean is not installed. Run: pixi run lean-setup")
    if (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise SystemExit("Install elan for your platform, then rerun lean-setup.")
    with tempfile.TemporaryDirectory(prefix="pymablock-elan-") as temporary:
        archive = Path(temporary) / "elan.tar.gz"
        urllib.request.urlretrieve(ELAN_URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != ELAN_SHA256:
            raise SystemExit("Elan archive checksum mismatch")
        with tarfile.open(archive) as bundle:
            member = bundle.getmember("elan-init")
            if not member.isfile():
                raise SystemExit("Expected an elan-init executable")
            installer = Path(temporary) / "elan-init"
            with bundle.extractfile(member) as source:
                installer.write_bytes(source.read())
        installer.chmod(0o755)
        subprocess.run(
            [str(installer), "-y", "--no-modify-path", "--default-toolchain", "none"],
            check=True,
        )
    return [str(lake)]


def run(lake: list[str], *arguments: str) -> None:
    """Run a Lake command in the formalization project."""
    subprocess.run([*lake, *arguments], cwd=PROJECT, check=True)


def report() -> None:
    """Render Lean's checked types; prose never supplies theorem statements."""
    catalog = json.loads((REPORTS / "correspondence.json").read_text())
    declarations = {row["name"]: row for row in catalog["declarations"]}
    registered = {(row["source"], row["label"]) for row in catalog["occurrences"]}
    labels = set()
    for source in sorted({source for source, _ in registered}):
        document = (ROOT / source).read_text()
        found = re.findall(r"\\label\{(eq:[^}]+)\}", document)
        found += re.findall(r"^:label:\s*(nh:\S+)", document, re.MULTILINE)
        labels.update((source, label) for label in found)
    if unknown := registered - labels:
        raise SystemExit(f"Registry labels absent from sources: {sorted(unknown)}")
    if missing := set(catalog["roots"]) - declarations.keys():
        raise SystemExit(f"Missing root declarations: {sorted(missing)}")
    gaps = [f"{source}#{label}" for source, label in sorted(labels - registered)]
    catalog["unregistered_equations"] = gaps
    (REPORTS / "correspondence.json").write_text(json.dumps(catalog, indent=2) + "\n")
    lines = [
        "# Pymablock formalization: checked correspondence report",
        "",
        "Generated from Lean's environment after compiling the proof library.",
        "The registry locates related manuscript and documentation equations; it does not prove that",
        "the manuscript text or Python source is identical to a Lean declaration.",
        "",
        f"- Terminal theorems: {len(catalog['roots'])}",
        f"- Reachable project declarations: {len(declarations)}",
        f"- Registered equation labels: {len(registered)}",
        "- Allowed foundational axioms: propext, Classical.choice, Quot.sound.",
        "- Export rejects sorryAx, custom axioms, and native compiler-oracle axioms.",
        "",
        "## Proved least-action minimality",
        "",
        "For any finite block partition, a unitary U with positive-definite diagonal",
        "blocks uniquely minimizes Frobenius distance to I among unitaries T with",
        "the same assigned subspaces: U P_a U-adjoint = T P_a T-adjoint for each block.",
        "Positive definite means Hermitian with strictly positive eigenvalues, not",
        "positive entries. The conclusion is ||U-I||_F <= ||T-I||_F, with equality iff T=U.",
        "No Hamiltonian or gap is needed for this geometric theorem; those enter the",
        "construction of the transformation and its assigned invariant subspaces.",
        "",
        "The proof derives T=UD with D block-diagonal and unitary, and checks the exact",
        "certificate ||UD-I||_F^2 - ||U-I||_F^2 = ||sqrt(A)(D-I)||_F^2, A=diag_blocks(U).",
        "This is the full-matrix form of the block sum; no block-label enumeration",
        "is required. Invertibility of sqrt(A) proves uniqueness.",
        "The explicit trace objective is proved equal to mathlib's Frobenius norm squared.",
        "",
        "The constructed formal output has Hermitian retained coefficients.",
        "Continuity through I and the gauge imply positive diagonal blocks locally.",
        "The convergence theorem below now supplies the continuous unitary realization",
        "from analytic finite-matrix input, rather than assuming it independently.",
        "Arbitrary selective masks, non-Hermitian transformations, and other norms",
        "are outside the minimality theorem.",
        "",
        "## Proved convergence and realization",
        "",
        "For finite complex matrices, any block partition, diagonal H0, cross-block",
        "energy separation, and Hermitian formal input, matrix_convergent proves a",
        "positive absolute convergence radius for both U and H_tilde.",
        "The additional input assumption is R>0 and sum_n ||H_n||_F R^|n| < infinity.",
        "Polynomial inputs satisfy it automatically. All mixed perturbation terms are",
        "included. Output convergence and a continuous realization are conclusions.",
        "",
        "The proof uniformly bounds finite iterations in an absolute weighted",
        "coefficient sum, then uses coefficient stabilization of the existing causal",
        "construction. For K>=1 bounding both projections and the Sylvester solver,",
        "t=1/(16 K^2) and input mass epsilon=t^2 give an invariant bound on q and B.",
        "The concrete finite-matrix maps supply K. Dominated convergence makes the",
        "positive-degree input mass small enough by shrinking the radius.",
        "The theorem proves existence of a positive radius, not its optimal value.",
        "",
        "Evaluation is proved absolutely summable and continuous on the polydisc.",
        "Cauchy products and adjoints pass through summation. The summed matrices",
        "satisfy both unitary identities, conjugation, block elimination, and the gauge.",
        "For any finite number of real parameters, matrix_least_action derives local",
        "unique Frobenius minimality directly from the analytic input assumptions.",
        "The former continuous-realization and realized-identity assumptions are thus",
        "discharged in this finite Hermitian block setting.",
        "The abstract estimate covers the Hermitian/selective recurrence with explicit",
        "bounded linear maps. The concrete theorem uses ordinary block partitions.",
        "Non-Hermitian convergence is not part of this result.",
        "",
        "## Correspondence boundaries and manuscript finding",
        "",
        "The proofs cover the Hermitian recurrence with arbitrary symmetric entry masks",
        "and the documented non-Hermitian recurrence with a tracked inverse. Retained",
        "entries may be degenerate; eliminated entries require",
        "distinct unperturbed energies. The retained mask need not be transitive.",
        "The non-Hermitian theorem permits complex energies and asymmetric masks.",
        "Its concrete matrix solver assumes an eigenbasis for H0. The generic theorem",
        "instead takes an explicit Sylvester solver contract. Biorthogonal basis",
        "construction and the implicit oblique-projector solver are not verified.",
        "The proofs do not verify the Python parser, caching, numerical solvers, floating",
        "point, or the optional",
        "two-block fast path or runtime complexity. Convergence has the analytic input",
        "and finite-matrix scope stated above. Recurrence uniqueness and geometric",
        "uniqueness theorem above have different premises.",
        "",
        "The printed optimized Sylvester equation (eq:sylvester_optimized) does not",
        "match the implementation: B - H' - A has the wrong leading sign and omits",
        "the Hermitian part of X. The checked recurrence uses",
        "[V,H0] = off(herm(B + H'_R + H'_R U') - [V,H'_S]).",
        "Already at first order the printed equation gives -H'_R instead of +H'_R.",
        "This report flags the discrepancy; the manuscript is not edited by this MR.",
        "",
        "## Equation registry",
        "",
        "| Source | Equation label | Lean declaration | Relation |",
        "| --- | --- | --- | --- |",
    ]
    for row in catalog["occurrences"]:
        lines.append(
            f"| {row['source']} | {row['label']} | `{row['declaration']}` | {row['relation']} |"
        )
    lines += ["", "Unregistered equation labels: " + ", ".join(gaps), ""]
    for name in catalog["roots"]:
        row = declarations[name]
        lines += [
            f"## {name}",
            "",
            "```lean",
            row["type"],
            "```",
            "",
            "Kernel axioms: " + ", ".join(row["axioms"]),
            "",
            "Proposition-valued inputs:",
            "",
        ]
        for hypothesis in row["hypotheses"]:
            lines.append(f"- `{hypothesis['name']}`: `{hypothesis['type']}`")
        if not row["hypotheses"]:
            lines.append("None; inspect the typed data inputs and definitions below.")
        lines.append("")
    lines += ["## Transitive definitions and assumptions", ""]
    for name, row in declarations.items():
        if row["kind"] not in {"definition", "inductive", "constructor"}:
            continue
        lines += [f"### {name}", "", "```lean", row["type"], "```", ""]
        if row["body"]:
            lines += ["```lean", row["body"], "```", ""]
    text = "\n".join(lines)
    (REPORTS / "correspondence.md").write_text(text)
    # A self-contained browser handoff retaining every checked signature.
    document = (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Pymablock Lean correspondence</title><style>"
        "body{max-width:1100px;margin:3rem auto;padding:0 1rem;color:#20252b;"
        "background:#fafafa;font:16px/1.6 system-ui}pre{white-space:pre-wrap;"
        "overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}"
        "table{border-collapse:collapse;width:100%;table-layout:fixed;font-size:14px}td,th{padding:.5rem;"
        "border:1px solid #ddd;text-align:left;overflow-wrap:anywhere}h2{margin-top:2.5rem;border-top:1px solid #ddd;"
        "padding-top:1.2rem}code{background:#eef1f4;padding:.1rem .2rem}pre{padding:1rem;"
        "background:#eef1f4}a{color:#155e75}</style>"
        + MarkdownIt("commonmark", {"html": False}).enable("table").render(text)
        + "</html>"
    )
    (REPORTS / "correspondence.html").write_text(document)
    print(f"Report: {REPORTS / 'correspondence.html'}")


def main() -> None:
    """Dispatch setup, validation, or report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "check", "report"])
    action = parser.parse_args().action
    lake = lake_command(install=action == "setup")
    if action == "setup":
        run(lake, "update")
        return
    run(lake, "build", "Pymablock", "Pymablock.Manuscript.Registry")
    if action == "check":
        for test in sorted((PROJECT / "Tests").glob("*.lean")):
            run(lake, "env", "lean", str(test.relative_to(PROJECT)))
    run(lake, "env", "lean", "Pymablock/Manuscript/Exports.lean")
    report()


if __name__ == "__main__":
    main()
