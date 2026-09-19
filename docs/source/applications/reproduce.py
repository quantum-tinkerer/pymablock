"""Execute an application document using the current Python environment.

Usage from the repository root:
    pixi run -e docs python docs/source/applications/reproduce.py supercurrent
"""

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import nbformat
from jupyter_client.kernelspec import KernelSpecManager
from myst_nb.core.read import read_myst_markdown_notebook
from nbclient import NotebookClient


def main():
    """Run selected documents and save notebooks with their computed outputs."""
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "documents",
        nargs="+",
        choices=sorted(
            path.stem for path in directory.glob("*.md") if path.stem != "index"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/build/applications")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = directory.parents[2]
    environment = {"PYTHONPATH": str(root)} if (root / "pymablock").is_dir() else {}
    with TemporaryDirectory(prefix="pymablock-kernel-") as temporary:
        kernel = Path(temporary, "python3")
        kernel.mkdir()
        (kernel / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "Pymablock applications",
                    "language": "python",
                    "env": environment,
                }
            )
        )
        manager = KernelSpecManager(kernel_dirs=[temporary])
        for name in args.documents:
            notebook = read_myst_markdown_notebook((directory / f"{name}.md").read_text())
            client = NotebookClient(
                notebook,
                timeout=480,
                kernel_name="python3",
                resources={"metadata": {"path": str(directory)}},
            )
            client.create_kernel_manager().kernel_spec_manager = manager
            client.execute()
            destination = args.output_dir / f"{name}.ipynb"
            nbformat.write(notebook, destination)
            print(f"Executed {name}: {destination}", flush=True)


if __name__ == "__main__":
    main()
