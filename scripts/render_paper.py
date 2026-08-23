#!/usr/bin/env python3
"""Render the audited Markdown paper to a reproducible, styled PDF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

_STYLE = """
@page { size: Letter; margin: 0.7in 0.75in; }
body { color: #171717; font: 10.2pt/1.47 Georgia, serif; margin: 0 auto;
       max-width: 7.1in; }
h1, h2, h3 { color: #111827; font-family: Arial, sans-serif;
             break-after: avoid; }
h1 { font-size: 22pt; line-height: 1.15; margin-bottom: 0.3em; }
h2 { border-bottom: 1px solid #d1d5db; font-size: 15pt; margin-top: 1.2em;
     padding-bottom: 0.15em; }
h3 { font-size: 11.5pt; }
p, li { orphans: 3; widows: 3; }
a { color: #075985; text-decoration: none; }
table { border-collapse: collapse; font: 8.7pt/1.3 Arial, sans-serif;
        margin: 0.7em 0; page-break-inside: avoid; width: 100%; }
th, td { border: 1px solid #cbd5e1; padding: 0.28em 0.38em; text-align: left;
         vertical-align: top; }
th { background: #f1f5f9; }
blockquote { border-left: 4px solid #64748b; color: #334155; margin: 0.8em 0;
             padding: 0.25em 0.75em; }
code { background: #f1f5f9; font: 8.8pt Consolas, monospace; padding: 0.05em 0.2em; }
pre { background: #f1f5f9; padding: 0.6em; white-space: pre-wrap; }
"""


def _html(markdown_text: str, *, title: str) -> str:
    body = markdown.markdown(
        markdown_text,
        extensions=("tables", "fenced_code", "sane_lists"),
        output_format="html5",
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{title}</title><style>{_STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/18-paper.md"))
    parser.add_argument("--output", type=Path, default=Path("docs/18-paper.pdf"))
    parser.add_argument("--chromium")
    args = parser.parse_args()

    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    chromium = args.chromium or shutil.which("chromium")
    if chromium is None and Path("/snap/bin/chromium").is_file():
        chromium = "/snap/bin/chromium"
    if chromium is None:
        parser.error("Chromium is required to render the PDF")

    output.parent.mkdir(parents=True, exist_ok=True)
    title = next(
        (
            line.removeprefix("# ").strip()
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.startswith("# ")
        ),
        source.stem,
    )
    document = _html(source.read_text(encoding="utf-8"), title=title)
    with tempfile.TemporaryDirectory(prefix="homymoly-paper-") as temporary:
        html_path = Path(temporary) / "paper.html"
        html_path.write_text(document, encoding="utf-8")
        subprocess.run(
            (
                chromium,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--no-pdf-header-footer",
                f"--print-to-pdf={output}",
                html_path.as_uri(),
            ),
            check=True,
            timeout=120,
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
