"""
Report generation utilities for benchmark runs.

The report builder consumes the structured ``run.json`` emitted by
``BenchmarkRun`` and renders lightweight Markdown/HTML summaries with
tables, configuration dumps, and linked artifacts.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..benchmark import load_run


def _format_table(rows: List[List[str]]) -> str:
    header, *body = rows
    header_line = " | ".join(header)
    separator = " | ".join(["---"] * len(header))
    body_lines = [" | ".join(r) for r in body]
    return "\n".join([header_line, separator, *body_lines])


def render_markdown(run_data: Dict[str, Any]) -> str:
    """Create a Markdown report for a benchmark run."""

    lines = [f"# Benchmark Report: {run_data['name']}", ""]
    lines.append(f"*Started at:* {run_data['started_at']}  ")
    lines.append(f"*Seed:* {run_data['seed']}  ")
    if run_data.get("metadata"):
        lines.append(f"*Tags:* {run_data['metadata']}  ")
    lines.append("")

    if run_data.get("config"):
        lines.append("## Configuration")
        config_rows = [["Key", "Value"]]
        for key, value in sorted(run_data["config"].items()):
            config_rows.append([str(key), f"`{value}`"])
        lines.append(_format_table(config_rows))
        lines.append("")

    if run_data.get("metrics"):
        lines.append("## Metrics")
        metric_rows = [["Metric", "Value"]]
        for key, value in sorted(run_data["metrics"].items()):
            metric_rows.append([key, f"`{value}`"])
        lines.append(_format_table(metric_rows))
        lines.append("")

    if run_data.get("timings"):
        lines.append("## Timings (seconds)")
        timing_rows = [["Section", "Duration"]]
        for key, value in sorted(run_data["timings"].items()):
            timing_rows.append([key, f"{value:.3f}"])
        lines.append(_format_table(timing_rows))
        lines.append("")

    if run_data.get("notes"):
        lines.append("## Notes")
        for note in run_data["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    if run_data.get("artifacts"):
        lines.append("## Artifacts")
        for art in run_data["artifacts"]:
            desc = art.get("description") or "artifact"
            path = art.get("path")
            lines.append(f"- **{desc}:** `{path}`")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_html(run_data: Dict[str, Any]) -> str:
    """Create a minimal HTML report with linked artifacts."""

    def esc(value: Any) -> str:
        return html.escape(str(value))

    rows = []
    for key, value in sorted(run_data.get("metrics", {}).items()):
        rows.append(f"<tr><td>{esc(key)}</td><td>{esc(value)}</td></tr>")
    metrics_table = "\n".join(rows)

    timing_rows = []
    for key, value in sorted(run_data.get("timings", {}).items()):
        timing_rows.append(f"<tr><td>{esc(key)}</td><td>{value:.3f}</td></tr>")
    timings_table = "\n".join(timing_rows)

    artifact_lines = []
    for art in run_data.get("artifacts", []):
        desc = esc(art.get("description") or "artifact")
        path = esc(art.get("path"))
        artifact_lines.append(f"<li><code>{desc}</code>: <a href='{path}'>{path}</a></li>")
    artifact_list = "\n".join(artifact_lines)

    return f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Benchmark Report: {esc(run_data['name'])}</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 2rem; }}
      table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
      th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
      th {{ background: #f5f5f5; }}
      code {{ background: #f0f0f0; padding: 0.1rem 0.2rem; }}
    </style>
  </head>
  <body>
    <h1>Benchmark Report: {esc(run_data['name'])}</h1>
    <p><strong>Started:</strong> {esc(run_data['started_at'])}<br />
       <strong>Seed:</strong> {esc(run_data['seed'])}</p>
    <h2>Metrics</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      {metrics_table}
    </table>
    <h2>Timings (seconds)</h2>
    <table>
      <tr><th>Section</th><th>Duration</th></tr>
      {timings_table}
    </table>
    <h2>Artifacts</h2>
    <ul>
      {artifact_list}
    </ul>
  </body>
</html>
"""


def save_report(
    run_dir: str | Path,
    *,
    run_data: Optional[Dict[str, Any]] = None,
    include_html: bool = True,
) -> Dict[str, str]:
    """Render Markdown/HTML reports for a benchmark run and save them."""

    run_data = run_data or load_run(run_dir)
    run_path = Path(run_dir)
    if run_path.is_file():
        run_path = run_path.parent

    run_path.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    md_content = render_markdown(run_data)
    md_path = run_path / "report.md"
    md_path.write_text(md_content, encoding="utf-8")
    paths["markdown"] = str(md_path)

    if include_html:
        html_content = render_html(run_data)
        html_path = run_path / "report.html"
        html_path.write_text(html_content, encoding="utf-8")
        paths["html"] = str(html_path)

    return paths

