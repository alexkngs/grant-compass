"""
Deterministic PI-discovery pipeline: OpenReview accepted papers -> last author
(flagged as likely PI) -> OpenAlex enrichment (institution, country, h-index) ->
filtered to Europe -> data/professors_raw.json.

Venues OpenReview doesn't host (design research venues such as Design Studies,
DRS, She Ji) are listed under search.openalex_venues by OpenAlex source id; their
papers come from OpenAlex's works endpoint, whose last authorship already names
the author's OpenAlex id, so enrichment is an exact id lookup, not a name search.

Clean-room design note: this reimplements the *concept* of arjunk00/phd-finder
(OpenReview -> last-author -> citation-metrics) using only OpenReview's and
OpenAlex's public, ToS-permitted REST APIs. No code from that repo (which
carries no license) was copied. OpenAlex expects a free API key once the shared
anonymous daily budget runs out; set OPENALEX_API_KEY.
"""
import json
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

from .config import load_config

OPENREVIEW_API = "https://api2.openreview.net/notes/search"
OPENREVIEW_MAX_LIMIT = 500  # larger limits come back empty
OPENALEX_API = "https://api.openalex.org/authors"
OPENALEX_WORKS_API = "https://api.openalex.org/works"
OPENALEX_MAX_PER_PAGE = 200
OPENALEX_KEY_ENV = "OPENALEX_API_KEY"
EUROPEAN_COUNTRY_CODES = {
    "DE", "FR", "NL", "CH", "SE", "NO", "FI", "DK", "IT", "ES", "PT", "AT",
    "BE", "IE", "PL", "GB", "UK", "CZ", "HU", "GR", "RO", "BG", "HR", "SI",
    "SK", "EE", "LV", "LT", "LU", "IS", "CY", "MT",
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class RateLimited(Exception):
    """Raised when an API answers 429, so callers can stop hammering it."""


_warned = set()


def _warn(key: str, message: str) -> None:
    """Print a warning once per key (e.g. once per failing endpoint)."""
    if key in _warned:
        return
    _warned.add(key)
    print(f"warning: {message}", file=sys.stderr)


def _get(url, params, headers=None, retries=2, timeout=15):
    """GET JSON, returning None on failure after warning once for this endpoint.

    Raises RateLimited on HTTP 429 instead of retrying.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 429:
                raise RateLimited(_error_detail(resp))
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries:
                detail = _error_detail(getattr(exc, "response", None)) or str(exc)
                _warn(url, f"{url} failed: {detail}")
                return None
            time.sleep(1)


def _error_detail(resp) -> str:
    if resp is None:
        return ""
    try:
        body = resp.json()
        msg = (body.get("message") or body.get("error") or body.get("name")) if isinstance(body, dict) else None
    except ValueError:
        msg = resp.text[:200]
    return f"HTTP {resp.status_code}" + (f" ({msg})" if msg else "")


def fetch_venue_papers(venue_keyword: str, max_papers: int) -> list[dict]:
    """Full-text search OpenReview for submissions (forums, not reviews) matching venue_keyword."""
    params = {
        "term": venue_keyword,
        "source": "forum",
        "limit": min(max_papers, OPENREVIEW_MAX_LIMIT),
    }
    data = _get(OPENREVIEW_API, params)
    if not data:
        return []
    return data.get("notes", [])


def _content_value(paper: dict, key: str):
    """Read a content field in either API v2 ({"value": x}) or v1 (plain x) shape."""
    field = (paper.get("content") or {}).get(key)
    return field.get("value") if isinstance(field, dict) else field


def paper_matches_venue(paper: dict, venue: str) -> bool:
    """True if the paper's venue/venueid names the venue acronym (case-sensitive, whole word).

    Full-text search also returns papers that merely mention the term (e.g. "DIS"
    as Dichotomous Image Segmentation), so filter on the venue fields.
    """
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(venue)}(?![A-Za-z])")
    for key in ("venue", "venueid"):
        value = _content_value(paper, key)
        if isinstance(value, str) and pattern.search(value):
            return True
    return False


def _name_from_profile_id(author_id: str) -> str | None:
    """Turn an OpenReview profile id like ~Jane_Q_Doe2 into "Jane Q Doe"; dblp URLs/emails -> None."""
    if not isinstance(author_id, str) or not author_id.startswith("~"):
        return None
    return re.sub(r"\d+$", "", author_id[1:]).replace("_", " ").strip() or None


def last_author_of(paper: dict) -> str | None:
    """Last author's name, or None for notes without authors (reviews, comments)."""
    authors = _content_value(paper, "authors")
    if isinstance(authors, list) and authors:
        last = authors[-1]
        if isinstance(last, str) and last.strip():
            return last.strip()
    author_ids = _content_value(paper, "authorids")
    if isinstance(author_ids, list) and author_ids:
        return _name_from_profile_id(author_ids[-1])
    return None


def _openalex_headers() -> dict:
    """Auth header from OPENALEX_API_KEY if set (never hardcode the key)."""
    key = os.environ.get(OPENALEX_KEY_ENV, "").strip()
    return {"Authorization": f"Bearer {key}"} if key else {}


def fetch_openalex_venue_works(source_id: str, keyword: str, years: list, max_papers: int) -> list[dict]:
    """Search one OpenAlex source (journal/conference) for works matching keyword. Raises RateLimited on 429."""
    source_id = source_id.rsplit("/", 1)[-1]
    filters = [f"primary_location.source.id:{source_id}"]
    if years:
        filters.append("publication_year:" + "|".join(str(y) for y in years))
    params = {
        "filter": ",".join(filters),
        "per_page": min(max_papers, OPENALEX_MAX_PER_PAGE),
        "select": "title,authorships",
    }
    if keyword:
        params["search"] = keyword
    data = _get(OPENALEX_WORKS_API, params, headers=_openalex_headers())
    if not data:
        return []
    return data.get("results", [])


def last_authorship_of(work: dict) -> dict | None:
    """Last author of an OpenAlex work as {"name", "id"}, or None if it has no identified authors."""
    authorships = work.get("authorships") or []
    if not authorships:
        return None
    author = authorships[-1].get("author") or {}
    if not author.get("id") or not author.get("display_name"):
        return None
    return {"name": author["display_name"], "id": author["id"]}


def enrich_author_by_id(author_id: str) -> dict | None:
    """Fetch one OpenAlex author by id. Raises RateLimited on HTTP 429."""
    a = _get(f"{OPENALEX_API}/{author_id.rsplit('/', 1)[-1]}", {}, headers=_openalex_headers())
    return _author_record(a, a.get("display_name")) if a else None


def enrich_author(name: str) -> dict | None:
    """Look up name on OpenAlex. Raises RateLimited on HTTP 429."""
    data = _get(OPENALEX_API, {"search": name, "per_page": 1}, headers=_openalex_headers())
    if not data or not data.get("results"):
        return None
    return _author_record(data["results"][0], name)


def _author_record(a: dict, name: str) -> dict:
    inst = (a.get("last_known_institutions") or [{}])[0]
    country = inst.get("country_code")
    return {
        "name": a.get("display_name", name),
        "institution": inst.get("display_name"),
        "country_code": country,
        "h_index": (a.get("summary_stats") or {}).get("h_index"),
        "works_count": a.get("works_count"),
        "openalex_id": a.get("id"),
    }


def _paper_candidates(search_cfg: dict, keywords: list):
    """Yield (venue, author_name, openalex_author_id | None, paper_title) across both paper sources."""
    max_papers = search_cfg.get("max_papers_per_venue", 50)
    for venue in search_cfg.get("venues") or []:
        for kw in keywords or [venue]:
            term = f"{venue} {kw}".strip()
            for paper in fetch_venue_papers(term, max_papers):
                if paper_matches_venue(paper, venue):
                    yield venue, last_author_of(paper), None, _content_value(paper, "title")
    for venue in search_cfg.get("openalex_venues") or []:
        for kw in keywords or [""]:
            for work in fetch_openalex_venue_works(venue["id"], kw, search_cfg.get("years") or [], max_papers):
                author = last_authorship_of(work)
                if author:
                    yield venue["name"], author["name"], author["id"], work.get("title")


def run() -> list[dict]:
    cfg = load_config()
    search_cfg = cfg.get("search", {})
    keywords = cfg.get("applicant", {}).get("field_keywords", [])

    records = []
    seen_authors = set()
    candidates = 0
    try:
        for venue, author, author_id, title in _paper_candidates(search_cfg, keywords):
            if not author or (author_id or author) in seen_authors:
                continue
            seen_authors.add(author_id or author)
            candidates += 1
            enriched = enrich_author_by_id(author_id) if author_id else enrich_author(author)
            if not enriched:
                continue
            if enriched.get("country_code") not in EUROPEAN_COUNTRY_CODES:
                continue
            enriched["source_venue"] = venue
            enriched["source_paper_title"] = title
            records.append(enriched)
    except RateLimited as exc:
        # The key may also be injected by a proxy, so only hint about it once OpenAlex refuses.
        hint = "" if os.environ.get(OPENALEX_KEY_ENV) else (
            f" Set {OPENALEX_KEY_ENV} (free key: https://help.openalex.org/api/authentication/).")
        _warn("openalex-429", f"OpenAlex rate limit hit, stopping enrichment: {exc}.{hint}")

    if not candidates:
        venues = search_cfg.get("venues", []) + [v["name"] for v in search_cfg.get("openalex_venues") or []]
        _warn("no-candidates", f"no papers with authors matched venues {venues} "
              f"and keywords {keywords}")

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "professors_raw.json"
    out_path.write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    results = run()
    print(f"Wrote {len(results)} European PI records to data/professors_raw.json")
