"""Fetch intersphinx inventories before Sphinx starts.

Sphinx treats missing intersphinx targets as documentation warnings. Fetching
inventories outside Sphinx lets CI retry live downloads, preserve a cache, and
fall back to committed emergency inventories without hiding real documentation
warnings.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.environ.get("PYMABLOCK_INTERSPHINX_DIR", ROOT / "docs/build/intersphinx")
)
CACHE_DIR = Path(
    os.environ.get("PYMABLOCK_INTERSPHINX_CACHE", ROOT / ".intersphinx-cache")
)
FALLBACK_DIR = ROOT / "docs/source/_static/intersphinx-fallback"
TIMEOUT = float(os.environ.get("PYMABLOCK_INTERSPHINX_TIMEOUT", "20"))
ATTEMPTS = int(os.environ.get("PYMABLOCK_INTERSPHINX_ATTEMPTS", "3"))

INVENTORIES = {
    "python": "https://docs.python.org/3/objects.inv",
    "kwant": "https://kwant-project.org/doc/1/objects.inv",
    "numpy": "https://numpy.org/doc/stable/objects.inv",
    "scipy": "https://docs.scipy.org/doc/scipy/objects.inv",
    # TODO: Switch to latest when sympy 1.15 is released.
    "sympy": "https://docs.sympy.org/dev/objects.inv",
}


def _is_inventory(path: Path) -> bool:
    try:
        with path.open("rb") as file:
            return file.readline().startswith(b"# Sphinx inventory version")
    except OSError:
        return False


def _fetch(url: str, destination: Path) -> None:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "pymablock intersphinx inventory prefetch "
                "(https://gitlab.kwant-project.org/qt/pymablock)"
            )
        },
    )
    temporary = destination.with_suffix(".tmp")
    with urlopen(request, timeout=TIMEOUT) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    if not _is_inventory(temporary):
        temporary.unlink(missing_ok=True)
        raise ValueError(f"downloaded file is not a Sphinx inventory: {url}")
    temporary.replace(destination)


def _copy_valid(source: Path, destination: Path) -> bool:
    if not _is_inventory(source):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _fetch_with_retries(name: str, url: str, destination: Path) -> bool:
    for attempt in range(1, ATTEMPTS + 1):
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _fetch(url, destination)
            print(f"intersphinx {name}: live -> {destination}")
            return True
        except (OSError, TimeoutError, URLError, ValueError) as exc:
            print(
                f"intersphinx {name}: live attempt {attempt}/{ATTEMPTS} failed: {exc}",
                file=sys.stderr,
            )
            if attempt < ATTEMPTS:
                time.sleep(attempt)
    return False


def main() -> int:
    """Fetch or copy every configured inventory."""
    failures: list[str] = []
    for name, url in INVENTORIES.items():
        destination = OUTPUT_DIR / f"{name}.inv"
        cached = CACHE_DIR / f"{name}.inv"
        fallback = FALLBACK_DIR / f"{name}.inv"

        if _fetch_with_retries(name, url, destination):
            cached.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, cached)
            continue
        if _copy_valid(cached, destination):
            print(f"intersphinx {name}: cache -> {destination}")
            continue
        if _copy_valid(fallback, destination):
            print(f"intersphinx {name}: emergency fallback -> {destination}")
            continue
        failures.append(name)

    if failures:
        print(f"missing intersphinx inventories: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
