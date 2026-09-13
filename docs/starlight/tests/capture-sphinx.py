"""Capture a semantic comparison fixture from the pre-migration Sphinx build.

Run with BeautifulSoup installed, passing that build's HTML directory.
The baseline commit and intentional exclusions are recorded in the fixture.
"""

import json
import re
import sys
import zlib
from pathlib import Path

from bs4 import BeautifulSoup

baseline = Path(sys.argv[1])
fixture = {
    "commit": "f85fcd1",
    "exclusions": [
        "Random UUID anchors assigned to unlabelled equations are not stable URLs.",
        "Sphinx's generic bool constructor text on dynamic SymPy properties is replaced by assumption-query descriptions.",
        "Signatures, annotation aliases and parameter metadata are checked separately from prose.",
    ],
    "pages": {},
    "objects": {},
}


def scrub(node):
    """Remove non-prose markup before comparing textual content."""
    for element in node.select(".math,pre,.headerlink"):
        element.decompose()


api = BeautifulSoup(
    (baseline / "documentation/pymablock.html").read_text(), "html.parser"
)
for sig in api.select("dt.sig[id]"):
    path = sig["id"]
    sibling = sig.find_next_sibling("dd")
    body = BeautifulSoup(str(sibling), "html.parser") if sibling else None
    prose = []
    if body:
        for nested in body.select("dl.py"):
            nested.decompose()
        scrub(body)
        for paragraph in body.select("p"):
            if paragraph.find("p") or paragraph.find_parent("dt"):
                continue
            text = paragraph.get_text(" ", strip=True)
            if text.startswith(
                "Returns True when the argument is true, False otherwise."
            ):
                continue
            # Napoleon puts parameter names/types before an en-dash; layouts
            # may relocate this metadata, but must preserve its description.
            if " \N{EN DASH} " in text:
                text = text.split(" \N{EN DASH} ", 1)[1]
            field = paragraph.find_parent("dd")
            label = field.find_previous_sibling("dt") if field else None
            if label and label.get_text(strip=True) in {"Return type:", "Type:"}:
                continue
            if len(text) > 30:
                prose.append(text)
    fixture["objects"][path] = prose

for file in sorted(baseline.rglob("*.html")):
    page = file.relative_to(baseline).with_suffix("").as_posix()
    source = Path("../source", page + ".md")
    if not source.exists() or page == "documentation/pymablock":
        continue
    article = BeautifulSoup(file.read_text(), "html.parser").select_one("article")
    equations = {
        node["id"]: node.select_one(".eqno")
        .get_text(" ", strip=True)
        .replace("#", "")
        .strip()
        for node in article.select(".math[id]")
        if node.select_one(".eqno")
    }
    for label, text in list(equations.items()):
        if re.fullmatch(r"equation-[0-9a-f-]{36}", label):
            del equations[label]
    scrub(article)
    paragraphs = [
        p.get_text(" ", strip=True)
        for p in article.select("p")
        if len(p.get_text(strip=True)) > 30
    ]
    fixture["pages"][page] = {"prose": paragraphs, "equations": equations}
payload = zlib.decompress(
    (baseline / "objects.inv").read_bytes().split(b"\n", 4)[4]
).decode()
fixture["inventory"] = [
    {"name": row[0], "type": row[1]}
    for line in payload.splitlines()
    if (row := line.split(None, 4))
]
Path("tests/fixtures/sphinx-parity.json").write_text(
    json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
)
