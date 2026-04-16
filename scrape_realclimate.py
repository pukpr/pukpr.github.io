#!/usr/bin/env python3
"""
scrape_realclimate.py
---------------------
Scrape RealClimate.org for comments posted by Paul Pukite (@whut) using the
WordPress REST API, then save the results as JSON and Markdown.

Usage
-----
    python scrape_realclimate.py [--output-dir OUTPUT_DIR] [--delay DELAY]

Dependencies
------------
    pip install requests

Output files
------------
    realclimate_whut_comments.json   – machine-readable list of comments
    realclimate_whut_comments.md     – human-readable Markdown document
"""

import argparse
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://www.realclimate.org"
API_BASE = f"{BASE_URL}/wp-json/wp/v2"

# Names / handles used by Paul Pukite on RealClimate
AUTHOR_NAMES = ["whut", "Paul Pukite", "paul pukite"]

PER_PAGE = 100  # WordPress REST API maximum
REQUEST_TIMEOUT = 30  # seconds

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (compatible; RealClimate-comment-scraper/1.0; "
                "+https://pukpr.github.io)"
            )
        }
    )
    return session


def fetch_comments_page(
    session: requests.Session,
    author_name: str,
    page: int,
    delay: float,
) -> list[dict]:
    """Fetch one page of comments filtered by author_name."""
    params = {
        "author_name": author_name,
        "per_page": PER_PAGE,
        "page": page,
        "orderby": "date",
        "order": "asc",
        "_fields": "id,date,link,post,content,author_name,author_url",
    }
    try:
        resp = session.get(
            f"{API_BASE}/comments",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 400:
            # WordPress returns 400 when the page is out of range
            return []
        resp.raise_for_status()
        time.sleep(delay)
        return resp.json()
    except requests.RequestException as exc:
        print(f"  [WARNING] Page {page} for '{author_name}': {exc}")
        return []


def fetch_post_url(session: requests.Session, post_id: int, delay: float) -> str:
    """Return the public URL of a post given its numeric ID."""
    if post_id == 0:
        return BASE_URL
    try:
        resp = session.get(
            f"{API_BASE}/posts/{post_id}",
            params={"_fields": "link,title"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        time.sleep(delay)
        return data.get("link", BASE_URL)
    except requests.RequestException as exc:
        print(f"  [WARNING] Could not resolve post {post_id}: {exc}")
        return BASE_URL


def collect_all_comments(session: requests.Session, delay: float) -> list[dict]:
    """Collect every comment from any known author handle, de-duplicated by id."""
    seen_ids: set[int] = set()
    comments: list[dict] = []

    for name in AUTHOR_NAMES:
        print(f"\nSearching for author name: '{name}' …")
        page = 1
        while True:
            print(f"  Fetching page {page} …", end=" ", flush=True)
            results = fetch_comments_page(session, name, page, delay)
            if not results:
                print("(no results / end of pages)")
                break
            new_count = 0
            for c in results:
                cid = c.get("id", 0)
                if cid not in seen_ids:
                    seen_ids.add(cid)
                    comments.append(c)
                    new_count += 1
            print(f"got {len(results)} → {new_count} new (total {len(comments)})")
            if len(results) < PER_PAGE:
                break
            page += 1

    return comments


def enrich_with_post_url(
    session: requests.Session, comments: list[dict], delay: float
) -> list[dict]:
    """Add a resolved post_url field by looking up each unique post id."""
    unique_post_ids: set[int] = {c.get("post", 0) for c in comments}
    print(f"\nResolving URLs for {len(unique_post_ids)} unique post(s) …")
    post_url_cache: dict[int, str] = {}
    for post_id in sorted(unique_post_ids):
        if post_id not in post_url_cache:
            url = fetch_post_url(session, post_id, delay)
            post_url_cache[post_id] = url
            print(f"  post {post_id} → {url}")

    for c in comments:
        post_id = c.get("post", 0)
        # Prefer the comment's own permalink if available
        c["post_url"] = c.get("link") or post_url_cache.get(post_id, BASE_URL)

    return comments


def plain_text(html: str) -> str:
    """Very light HTML → plain-text conversion (no external deps)."""
    import re

    text = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#039;": "'",
        "&nbsp;": " ",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    return text.strip()


def save_json(comments: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(comments, fh, indent=2, ensure_ascii=False)
    print(f"Saved JSON  → {path}")


def save_markdown(comments: list[dict], path: Path) -> None:
    lines: list[str] = [
        "# RealClimate.org — Comments by Paul Pukite (@whut)",
        "",
        f"*{len(comments)} comment(s) found.*",
        "",
        "---",
        "",
    ]
    for i, c in enumerate(comments, start=1):
        date = c.get("date", "unknown date")
        url = c.get("post_url", BASE_URL)
        body = plain_text(c.get("content", {}).get("rendered", ""))
        lines += [
            f"## Comment {i}",
            "",
            f"**Date:** {date}  ",
            f"**URL:** <{url}>",
            "",
            body,
            "",
            "---",
            "",
        ]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Saved Markdown → {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape RealClimate.org comments by Paul Pukite (@whut)."
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output files (default: current directory).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds to pause between HTTP requests (default: 1.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    comments = collect_all_comments(session, args.delay)
    if not comments:
        print("\nNo comments found. The WordPress REST API may not expose them.")
        return

    comments = enrich_with_post_url(session, comments, args.delay)

    # Sort chronologically
    comments.sort(key=lambda c: c.get("date", ""))

    json_path = output_dir / "realclimate_whut_comments.json"
    md_path = output_dir / "realclimate_whut_comments.md"

    save_json(comments, json_path)
    save_markdown(comments, md_path)

    print(f"\nDone. {len(comments)} comment(s) saved.")


if __name__ == "__main__":
    main()
