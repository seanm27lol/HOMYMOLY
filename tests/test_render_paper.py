from __future__ import annotations

from scripts.render_paper import _html


def test_paper_html_renders_tables_and_title() -> None:
    rendered = _html("# Title\n\n| A | B |\n|---|---|\n| 1 | 2 |", title="Title")
    assert "<title>Title</title>" in rendered
    assert "<table>" in rendered
    assert "<h1>Title</h1>" in rendered
