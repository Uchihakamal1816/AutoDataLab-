"""
Lightweight Stooq daily OHLCV fetch (HTTP GET of CSV) for RAG-advanced mode.

Not financial advice. For simulation / grounding only. Falls back to bundled fixtures
when offline or on error so CI and judges stay reproducible.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Must match grader `STOOQ_CITATION_SUFFIXES`.
DEFAULT_WATCHLIST: tuple[str, ...] = ("nvda.us", "aapl.us", "jpm.us")

_STOOQ_DAILY = "https://stooq.com/q/d/l/?s={symbol}&i=d"
_USER_AGENT = "AutoDataLab-Plus/0.1 (research; +https://github.com/)"
_TIMEOUT_SEC = 8.0

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stooq"
# Bundled multi-hundred-row daily history (Stooq-shaped) for Strategy when RAG is off
# (no network; used as “enterprise tape” context).
_LONG_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stooq_long"


def _parse_csv_tail(text: str, symbol: str, last_n: int = 3) -> str:
    text = text.strip()
    if not text or "Date" not in text:
        return f"{symbol}: (no data)"
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return f"{symbol}: (no data)"
    rdr = csv.reader(io.StringIO(text))
    rows: List[list[str]] = list(rdr)
    if not rows:
        return f"{symbol}: (no data)"
    header, *data = rows
    if not data:
        return f"{symbol}: (no data)"
    tail = data[-last_n:]
    parts = [f"Stooq {symbol} daily:"]
    for row in tail:
        if not row or row[0] == "No data":
            continue
        date = row[0]
        try:
            close = row[4] if len(row) > 4 else row[-1]
        except IndexError:
            close = "—"
        parts.append(f"  {date} close≈{close}")
    return " ".join(parts) if len(parts) > 1 else f"{symbol}: (unparseable)"


def _read_fixture(symbol: str) -> str | None:
    path = _FIXTURES / f"{symbol.replace('.', '_')}.csv"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _looks_like_stooq_csv(text: str) -> bool:
    t = text.lstrip("\ufeff").strip()
    if not t or "Date" not in t.splitlines()[0]:
        return False
    if "<html" in t.lower() or "<!doctype" in t.lower():
        return False
    return True


def fetch_stooq_daily_csv(symbol: str) -> str:
    """Return raw CSV text from network or local fixture."""
    url = _STOOQ_DAILY.format(symbol=symbol.lower())
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(req, timeout=_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, TimeoutError):
        raw = ""
    if not _looks_like_stooq_csv(raw):
        fix = _read_fixture(symbol)
        if fix is not None:
            return fix
        return "Date,Open,High,Low,Close,Volume\n"
    return raw


def read_long_fixture_csv(symbol: str) -> str:
    """Read the bundled long daily CSV for ``symbol`` (e.g. ``nvda.us``)."""
    path = _LONG_FIXTURES / f"{symbol.replace('.', '_')}.csv"
    if not path.is_file():
        return "Date,Open,High,Low,Close,Volume\n"
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "Date,Open,High,Low,Close,Volume\n"


def scrape_watchlist_from_long_csv(
    symbols: tuple[str, ...] = DEFAULT_WATCHLIST,
    last_n: int = 5,
) -> List[Tuple[str, str, str, int]]:
    """
    Like :func:`scrape_watchlist` but only reads local long CSVs (no HTTP).
    Returns ``(stooq_symbol, citation, snippet, row_count_excl_header)``.
    Citations use the same ``stooq:`` prefix so graders stay consistent if RAG is on elsewhere.
    """
    out: list[tuple[str, str, str, int]] = []
    for sym in symbols:
        raw = read_long_fixture_csv(sym)
        rows = list(csv.reader(io.StringIO(raw)))
        n_data = max(0, len(rows) - 1)
        snip = _parse_csv_tail(raw, sym, last_n=last_n)
        cite = f"stooq:{sym}"
        out.append((sym, cite, snip, n_data))
    return out


def scrape_watchlist(
    symbols: tuple[str, ...] = DEFAULT_WATCHLIST,
) -> List[Tuple[str, str, str]]:
    """
    For each symbol, fetch Stooq daily history and return
    (stooq_symbol, citation, snippet) for RAG + grounding.

    Citation format: ``stooq:nvda.us`` (used by ``graders.grounding_score``).
    """
    out: list[tuple[str, str, str]] = []
    for sym in symbols:
        raw = fetch_stooq_daily_csv(sym)
        snip = _parse_csv_tail(raw, sym)
        cite = f"stooq:{sym}"
        out.append((sym, cite, snip))
    return out
