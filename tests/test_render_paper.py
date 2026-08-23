from __future__ import annotations

from scripts.render_paper import _html, _pdf_page_count, _render_problem


def test_paper_html_renders_tables_and_title() -> None:
    rendered = _html("# Title\n\n| A | B |\n|---|---|\n| 1 | 2 |", title="Title")
    assert "<title>Title</title>" in rendered
    assert "<table>" in rendered
    assert "<h1>Title</h1>" in rendered


def test_figures_and_code_blocks_are_kept_whole_across_page_breaks() -> None:
    assert "page-break-inside: avoid" in _html("# T", title="T").split("pre {")[1]


def test_page_count_ignores_the_page_tree_node() -> None:
    pdf = b"%PDF-1.4\n/Type /Pages\n/Type /Page\n/Type /Page\n"
    assert _pdf_page_count(pdf) == 2


def test_render_problem_flags_a_missing_or_non_pdf_output(tmp_path) -> None:
    missing = tmp_path / "absent.pdf"
    assert _render_problem(missing, "x" * 100) == "no output file was produced"

    not_a_pdf = tmp_path / "bad.pdf"
    not_a_pdf.write_bytes(b"<html>error</html>")
    assert _render_problem(not_a_pdf, "x" * 100) == "output is not a PDF"


def test_render_problem_flags_a_long_document_collapsed_to_one_page(tmp_path) -> None:
    """A snap-confined Chromium renders its own error page and still exits zero."""

    output = tmp_path / "paper.pdf"
    output.write_bytes(b"%PDF-1.4\n/Type /Pages\n/Type /Page\n")
    problem = _render_problem(output, "x" * 180_000)
    assert problem is not None
    assert "rendered 1 page(s)" in problem


def test_render_problem_accepts_a_plausible_render(tmp_path) -> None:
    output = tmp_path / "paper.pdf"
    output.write_bytes(b"%PDF-1.4\n/Type /Pages\n" + b"/Type /Page\n" * 15)
    assert _render_problem(output, "x" * 180_000) is None
