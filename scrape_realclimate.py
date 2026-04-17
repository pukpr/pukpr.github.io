#!/usr/bin/env python3
"""
scrape_realclimate.py
---------------------
Scrape RealClimate comments for an exact author match and save the results as
JSON and Markdown.

Why this script exists
----------------------
The WordPress comments endpoint at RealClimate appears to ignore the
`author_name` query parameter, so server-side filtering can silently return
comments from unrelated authors. This scraper avoids that bug by collecting
candidate comments and filtering them client-side using exact normalized author
names.

Usage
-----
Fast search-based mode (recommended first):
    python3 scrape_realclimate.py \
        --author "Paul Pukite (@whut)" \
        --search-term whut \
        --search-term pukite

Exhaustive full scan (slower, but more robust if aliases changed over time):
    python3 scrape_realclimate.py \
        --scan-mode full-scan \
        --author "Paul Pukite (@whut)"

Dependencies
------------
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import time
from html import unescape
from pathlib import Path

import requests

BASE_URL = "https://www.realclimate.org"
API_BASE = f"{BASE_URL}/wp-json/wp/v2"
COMMENTS_ENDPOINT = f"{API_BASE}/comments"
POSTS_ENDPOINT = f"{API_BASE}/posts"

PER_PAGE = 100
REQUEST_TIMEOUT = 30
COMMENT_FIELDS = "id,date,link,post,content,author_name,author_url"

DEFAULT_AUTHOR_ALIASES = [
    "Paul Pukite (@whut)",
    "Paul Pukite",
    "whut",
]
DEFAULT_SEARCH_TERMS = [
    "whut",
    "pukite",
]


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; RealClimate-exact-author-scraper/1.0; "
                "+https://pukpr.github.io)"
            )
        }
    )
    return session


def fetch_comments_page(
    session: requests.Session,
    *,
    page: int,
    delay: float,
    search: str | None = None,
    order: str = "asc",
) -> tuple[list[dict], int]:
    params = {
        "per_page": PER_PAGE,
        "page": page,
        "orderby": "date",
        "order": order,
        "_fields": COMMENT_FIELDS,
    }
    if search:
        params["search"] = search

    try:
        resp = session.get(COMMENTS_ENDPOINT, params=params, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 400:
            return [], 0
        resp.raise_for_status()
        total_pages = int(resp.headers.get("X-WP-TotalPages", "0") or 0)
        time.sleep(delay)
        return resp.json(), total_pages
    except requests.RequestException as exc:
        label = f"search={search!r}" if search else "full scan"
        print(f"  [WARNING] Failed page {page} for {label}: {exc}")
        return [], 0


def fetch_post_url(session: requests.Session, post_id: int, delay: float) -> str:
    if post_id == 0:
        return BASE_URL

    try:
        resp = session.get(
            f"{POSTS_ENDPOINT}/{post_id}",
            params={"_fields": "link"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        time.sleep(delay)
        return data.get("link", BASE_URL)
    except requests.RequestException as exc:
        print(f"  [WARNING] Could not resolve post {post_id}: {exc}")
        return BASE_URL


def author_matches(author_name: str, author_aliases: set[str]) -> bool:
    return normalize_text(author_name) in author_aliases


def collect_search_candidates(
    session: requests.Session,
    *,
    search_terms: list[str],
    delay: float,
    max_pages: int | None,
) -> list[dict]:
    candidates: list[dict] = []
    seen_ids: set[int] = set()

    for term in search_terms:
        print(f"\nSearching comments for term: {term!r}")
        page = 1
        total_pages = None

        while True:
            if max_pages is not None and page > max_pages:
                break

            results, page_count = fetch_comments_page(
                session,
                page=page,
                delay=delay,
                search=term,
                order="asc",
            )
            if total_pages is None:
                total_pages = page_count

            if not results:
                print(f"  page {page}: no results / end")
                break

            new_count = 0
            for comment in results:
                comment_id = comment.get("id", 0)
                if comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    candidates.append(comment)
                    new_count += 1

            total_label = total_pages if total_pages else "?"
            print(
                f"  page {page}/{total_label}: got {len(results)} "
                f"-> {new_count} new candidate(s)"
            )

            if total_pages and page >= total_pages:
                break
            page += 1

    return candidates


def collect_full_scan_candidates(
    session: requests.Session,
    *,
    delay: float,
    max_pages: int | None,
) -> list[dict]:
    candidates: list[dict] = []
    page = 1
    total_pages = None

    print("\nRunning full comment scan (newest first) ...")
    while True:
        if max_pages is not None and page > max_pages:
            break

        results, page_count = fetch_comments_page(
            session,
            page=page,
            delay=delay,
            order="desc",
        )
        if total_pages is None:
            total_pages = page_count

        if not results:
            print(f"  page {page}: no results / end")
            break

        candidates.extend(results)
        total_label = total_pages if total_pages else "?"
        print(f"  page {page}/{total_label}: got {len(results)} comment(s)")

        if total_pages and page >= total_pages:
            break
        page += 1

    return candidates


def filter_comments_by_author(
    comments: list[dict], author_aliases: list[str]
) -> list[dict]:
    alias_set = {normalize_text(alias) for alias in author_aliases}
    filtered = [
        comment
        for comment in comments
        if author_matches(comment.get("author_name", ""), alias_set)
    ]
    filtered.sort(key=lambda comment: comment.get("date", ""))
    return filtered


def enrich_with_post_url(
    session: requests.Session, comments: list[dict], delay: float
) -> list[dict]:
    unique_post_ids = {comment.get("post", 0) for comment in comments}
    print(f"\nResolving URLs for {len(unique_post_ids)} unique post(s) ...")
    post_url_cache: dict[int, str] = {}

    for post_id in sorted(unique_post_ids):
        if post_id not in post_url_cache:
            post_url_cache[post_id] = fetch_post_url(session, post_id, delay)

    for comment in comments:
        post_id = comment.get("post", 0)
        comment["post_url"] = comment.get("link") or post_url_cache.get(post_id, BASE_URL)

    return comments


def plain_text(html: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", "", html)
    return unescape(text).strip()


def save_json(comments: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(comments, handle, indent=2, ensure_ascii=False)
    print(f"Saved JSON      -> {path}")


def save_markdown(comments: list[dict], path: Path) -> None:
    lines = [
        "# RealClimate.org — Exact-author comments",
        "",
        f"*{len(comments)} comment(s) found.*",
        "",
        "*Sections are keyed by the original RealClimate comment ID so other docs can link to them reliably.*",
        "",
        "---",
        "",
    ]

    for index, comment in enumerate(comments, start=1):
        comment_id = comment.get("id", 0)
        author_name = comment.get("author_name", "unknown author")
        date = comment.get("date", "unknown date")
        url = comment.get("post_url", BASE_URL)
        body = plain_text(comment.get("content", {}).get("rendered", ""))
        lines.extend(
            [
                f'<a id="comment-{comment_id}"></a>',
                "",
                f"## Comment {comment_id}",
                "",
                f"**Archive index:** {index}  ",
                f"**Author:** {author_name}  ",
                f"**Date:** {date}  ",
                f"**URL:** <{url}>",
                "",
                body,
                "",
                "---",
                "",
            ]
        )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"Saved Markdown  -> {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape RealClimate comments by exact author match.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause between HTTP requests.",
    )
    parser.add_argument(
        "--author",
        action="append",
        dest="authors",
        help=(
            "Exact author alias to match after normalization. "
            "Repeat as needed. If omitted, built-in Paul/whut aliases are used."
        ),
    )
    parser.add_argument(
        "--search-term",
        action="append",
        dest="search_terms",
        help=(
            "Search term used to gather candidate comments in search mode. "
            "Repeat as needed."
        ),
    )
    parser.add_argument(
        "--scan-mode",
        choices=["search", "full-scan"],
        default="search",
        help="Whether to gather candidates via search terms or by scanning all comments.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional page limit for debugging or partial runs.",
    )
    parser.add_argument(
        "--json-name",
        default="realclimate_exact_author_comments.json",
        help="Output JSON filename.",
    )
    parser.add_argument(
        "--markdown-name",
        default="realclimate_exact_author_comments.md",
        help="Output Markdown filename.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    author_aliases = args.authors or DEFAULT_AUTHOR_ALIASES
    search_terms = args.search_terms or DEFAULT_SEARCH_TERMS

    print("Author aliases:")
    for alias in author_aliases:
        print(f"  - {alias}")

    session = build_session()

    if args.scan_mode == "search":
        candidates = collect_search_candidates(
            session,
            search_terms=search_terms,
            delay=args.delay,
            max_pages=args.max_pages,
        )
    else:
        candidates = collect_full_scan_candidates(
            session,
            delay=args.delay,
            max_pages=args.max_pages,
        )

    print(f"\nCollected {len(candidates)} candidate comment(s).")
    comments = filter_comments_by_author(candidates, author_aliases)
    print(f"Matched {len(comments)} exact-author comment(s).")

    if not comments:
        print(
            "\nNo exact-author comments matched. "
            "Try additional --author aliases or --scan-mode full-scan."
        )
        return

    comments = enrich_with_post_url(session, comments, args.delay)

    json_path = output_dir / args.json_name
    markdown_path = output_dir / args.markdown_name
    save_json(comments, json_path)
    save_markdown(comments, markdown_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
