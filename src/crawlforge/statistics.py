"""Advanced crawl statistics and dependency-free report rendering."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from html import escape
from pathlib import Path
from time import perf_counter
from typing import TypedDict

from crawlforge.urls import canonical_hostname


class DomainCount(TypedDict):
    """Page count for one crawled domain."""

    domain: str
    pages: int


class CrawlerStatsSnapshot(TypedDict):
    """Serializable snapshot of advanced crawl metrics."""

    total_pages: int
    successful: int
    failed: int
    average_speed: float
    status_codes: dict[str, int]
    top_domains: list[DomainCount]
    elapsed_seconds: float
    progress_percent: float
    estimated_remaining_seconds: float | None
    active_tasks: int
    queued_pages: int


class CrawlerStats:
    """Collect page outcomes and derive live progress metrics."""

    def __init__(self, *, clock: Callable[[], float] = perf_counter) -> None:
        """Initialize an empty statistics collector."""
        self._clock = clock
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._target_pages = 0
        self._successful = 0
        self._failed = 0
        self._status_codes: Counter[int] = Counter()
        self._domains: Counter[str] = Counter()

    def reset(self, *, target_pages: int) -> None:
        """Start a fresh measurement window for one crawl."""
        if target_pages <= 0:
            raise ValueError("target_pages must be greater than zero")
        self._started_at = self._clock()
        self._finished_at = None
        self._target_pages = target_pages
        self._successful = 0
        self._failed = 0
        self._status_codes.clear()
        self._domains.clear()

    def record_page(
        self,
        url: str,
        *,
        successful: bool,
        status_code: int | None,
    ) -> None:
        """Record one final page outcome."""
        if self._started_at is None:
            raise RuntimeError("statistics collection has not started")
        if successful:
            self._successful += 1
        else:
            self._failed += 1
        if status_code is not None:
            self._status_codes[status_code] += 1
        hostname = canonical_hostname(url)
        if hostname:
            self._domains[hostname] += 1

    def finish(self) -> None:
        """Freeze elapsed time when a crawl finishes normally."""
        if self._started_at is not None and self._finished_at is None:
            self._finished_at = self._clock()

    def get_stats(
        self,
        *,
        active_tasks: int = 0,
        queued_pages: int = 0,
    ) -> CrawlerStatsSnapshot:
        """Return a serializable snapshot including speed and ETA."""
        if active_tasks < 0:
            raise ValueError("active_tasks must be zero or greater")
        if queued_pages < 0:
            raise ValueError("queued_pages must be zero or greater")

        total_pages = self._successful + self._failed
        elapsed = self._elapsed()
        average_speed = total_pages / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self._target_pages - total_pages)
        estimated_remaining = (
            remaining / average_speed if average_speed > 0 and remaining else None
        )
        if self._finished_at is not None:
            progress = 100.0
            estimated_remaining = None
        elif self._target_pages:
            progress = min(100.0, total_pages / self._target_pages * 100)
        else:
            progress = 0.0

        return {
            "total_pages": total_pages,
            "successful": self._successful,
            "failed": self._failed,
            "average_speed": average_speed,
            "status_codes": {
                str(status): count
                for status, count in sorted(self._status_codes.items())
            },
            "top_domains": [
                {"domain": domain, "pages": pages}
                for domain, pages in self._domains.most_common(10)
            ],
            "elapsed_seconds": elapsed,
            "progress_percent": progress,
            "estimated_remaining_seconds": estimated_remaining,
            "active_tasks": active_tasks,
            "queued_pages": queued_pages,
        }

    def export_to_json(self, filename: str | Path) -> None:
        """Export the current statistics snapshot as formatted JSON."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.get_stats(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def export_to_html_report(self, filename: str | Path) -> None:
        """Export the current statistics as a standalone HTML report."""
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            render_html_report(self.get_stats()),
            encoding="utf-8",
        )

    def _elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        ended_at = self._finished_at if self._finished_at is not None else self._clock()
        return max(0.0, ended_at - self._started_at)


def render_html_report(
    stats: Mapping[str, object],
    *,
    pages: Sequence[Mapping[str, object]] = (),
    failed_urls: Mapping[str, str] | None = None,
) -> str:
    """Render statistics and result tables without external assets."""
    status_codes = _integer_mapping(stats.get("status_codes"))
    domains = _domain_rows(stats.get("top_domains"))
    failures = failed_urls or {}
    status_max = max(status_codes.values(), default=1)
    domain_max = max((pages_count for _domain, pages_count in domains), default=1)

    status_rows = "".join(
        _bar_row(str(status), count, status_max)
        for status, count in status_codes.items()
    )
    domain_rows = "".join(
        _bar_row(domain, count, domain_max) for domain, count in domains
    )
    page_rows = "".join(
        "<tr>"
        f"<td>{escape(str(page.get('url', '')))}</td>"
        f"<td>{escape(str(page.get('title', '')))}</td>"
        "</tr>"
        for page in pages
    )
    failure_rows = "".join(
        f"<tr><td>{escape(url)}</td><td>{escape(error)}</td></tr>"
        for url, error in failures.items()
    )
    cards = (
        ("Total pages", stats.get("total_pages", 0)),
        ("Successful", stats.get("successful", 0)),
        ("Failed", stats.get("failed", 0)),
        ("Average speed", f"{_finite_number(stats.get('average_speed')):.2f} pages/s"),
        ("Elapsed", f"{_finite_number(stats.get('elapsed_seconds')):.2f} s"),
    )
    card_markup = "".join(
        f'<div class="card"><span>{escape(label)}</span>'
        f"<strong>{escape(str(value))}</strong></div>"
        for label, value in cards
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CrawlForge report</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, sans-serif; }}
    body {{ margin: 0 auto; max-width: 1100px; padding: 2rem; color: #172033; }}
    h1, h2 {{ margin-top: 0; }}
    section {{ margin: 2rem 0; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem;
    }}
    .card {{ background: #f4f7fb; border-radius: 10px; padding: 1rem; }}
    .card span {{ display: block; color: #56647a; font-size: .85rem; }}
    .card strong {{ display: block; font-size: 1.45rem; margin-top: .35rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{
      border-bottom: 1px solid #dfe5ee;
      padding: .6rem;
      text-align: left;
      vertical-align: top;
    }}
    .bar-track {{
      background: #e8edf5;
      border-radius: 999px;
      min-width: 180px;
      overflow: hidden;
    }}
    .bar {{ background: #3066e8; height: .75rem; min-width: 2px; }}
    code {{ overflow-wrap: anywhere; }}
  </style>
</head>
<body>
  <h1>CrawlForge report</h1>
  <div class="cards">{card_markup}</div>
  <section>
    <h2>Status codes</h2>
    <table><thead><tr><th>Status</th><th>Pages</th><th>Distribution</th></tr></thead>
    <tbody>{status_rows or _empty_row(3)}</tbody></table>
  </section>
  <section>
    <h2>Top domains</h2>
    <table><thead><tr><th>Domain</th><th>Pages</th><th>Distribution</th></tr></thead>
    <tbody>{domain_rows or _empty_row(3)}</tbody></table>
  </section>
  <section>
    <h2>Successful pages</h2>
    <table><thead><tr><th>URL</th><th>Title</th></tr></thead>
    <tbody>{page_rows or _empty_row(2)}</tbody></table>
  </section>
  <section>
    <h2>Failures</h2>
    <table><thead><tr><th>URL</th><th>Error</th></tr></thead>
    <tbody>{failure_rows or _empty_row(2)}</tbody></table>
  </section>
</body>
</html>
"""


def _bar_row(label: str, count: int, maximum: int) -> str:
    width = count / maximum * 100 if maximum else 0.0
    return (
        "<tr>"
        f"<td>{escape(label)}</td><td>{count}</td>"
        '<td><div class="bar-track">'
        f'<div class="bar" style="width:{width:.2f}%"></div>'
        "</div></td></tr>"
    )


def _empty_row(columns: int) -> str:
    return f'<tr><td colspan="{columns}">No data</td></tr>'


def _integer_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, int] = {}
    for key, item in value.items():
        if isinstance(key, str) and isinstance(item, int) and item >= 0:
            result[key] = item
    return result


def _domain_rows(value: object) -> list[tuple[str, int]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return []
    result: list[tuple[str, int]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        domain = item.get("domain")
        pages = item.get("pages")
        if isinstance(domain, str) and isinstance(pages, int) and pages >= 0:
            result.append((domain, pages))
    return result


def _finite_number(value: object) -> float:
    if isinstance(value, int | float):
        converted = float(value)
        if math.isfinite(converted):
            return converted
    return 0.0
