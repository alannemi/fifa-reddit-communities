"""
Fill gaps in live collection by fetching missed posts from Arctic Shift.

Scans data/collected/ and data/pending/ to find dates with no data,
then queries Arctic Shift for those dates and saves the posts with
their comments into data/collected/.
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.arctic_shift_client import ArcticShiftClient
from src.utils.storage import Storage

import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_existing_dates(data_dir: str) -> set:
    """Get all dates that have collected or pending data."""
    dates = set()
    collected_dir = os.path.join(data_dir, "collected")
    if os.path.exists(collected_dir):
        for d in os.listdir(collected_dir):
            if d.startswith("2026-"):
                dates.add(d)

    pending_dir = os.path.join(data_dir, "pending")
    if os.path.exists(pending_dir):
        for d in os.listdir(pending_dir):
            date_part = d[:10]
            if date_part.startswith("2026-"):
                dates.add(date_part)

    return dates


def get_existing_post_ids(data_dir: str) -> set:
    """Get all post IDs already collected or pending."""
    ids = set()
    for root, dirs, files in os.walk(os.path.join(data_dir, "collected")):
        for f in files:
            if f.startswith("post_") and f.endswith(".json"):
                ids.add(f.replace("post_", "").replace(".json", ""))
    for root, dirs, files in os.walk(os.path.join(data_dir, "pending")):
        for f in files:
            if f.startswith("post_") and f.endswith(".json"):
                ids.add(f.replace("post_", "").replace(".json", ""))
    return ids


def fill_date(client: ArcticShiftClient, date_str: str, storage: Storage,
              existing_ids: set, comment_limit: int = 10000,
              match_thread_limit: int = 2500):
    """Fetch all posts and comments for a given date from Arctic Shift."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    after_ts = int(dt.timestamp())
    before_ts = int((dt + timedelta(days=1)).timestamp())

    print(f"\n  Fetching posts for {date_str}...")
    posts = client.get_posts("soccer", after_ts, before_ts)
    print(f"  Found {len(posts)} posts")

    new_count = 0
    for post in posts:
        post_id = post.get("id", "")
        if post_id in existing_ids:
            continue

        is_match = any(kw in (post.get("title") or "").lower()
                       for kw in ["match thread", "post match", "pre match"])
        post["is_match_thread"] = is_match
        limit = match_thread_limit if is_match else comment_limit

        print(f"  Fetching comments for {post_id} ({post.get('title','')[:50]})")
        comments = client.get_comments_for_post("soccer", post_id)

        was_truncated = len(comments) > limit
        if was_truncated:
            comments = comments[:limit]

        post["t0_snapshot"] = {
            "score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
            "upvote_ratio": post.get("upvote_ratio"),
            "author_flair_text": post.get("author_flair_text"),
            "link_flair_text": post.get("link_flair_text"),
            "edited": post.get("edited"),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "batch_id": f"gap-fill-{date_str}",
        }
        post["t48_snapshot"] = post["t0_snapshot"].copy()
        post["comments"] = comments
        post["source"] = "arctic_shift"
        post["collection_metadata"] = {
            "total_comments_collected": len(comments),
            "comment_limit_applied": limit,
            "was_truncated": was_truncated,
            "gap_filled": True,
            "filled_at": datetime.now(timezone.utc).isoformat(),
        }
        post["media_urls"] = []

        date_dir = os.path.join(storage.collected_dir, date_str)
        os.makedirs(date_dir, exist_ok=True)
        filepath = os.path.join(date_dir, f"post_{post_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(post, f, indent=2, ensure_ascii=False)

        existing_ids.add(post_id)
        new_count += 1

    return new_count


def main():
    config = load_config()
    storage = Storage(config["storage"]["data_dir"])
    data_dir = config["storage"]["data_dir"]

    existing_dates = get_existing_dates(data_dir)
    existing_ids = get_existing_post_ids(data_dir)

    start = datetime(2026, 6, 10, tzinfo=timezone.utc)
    end = datetime(2026, 6, 21, tzinfo=timezone.utc)

    all_dates = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        all_dates.append(date_str)
        current += timedelta(days=1)

    gap_dates = [d for d in all_dates if d not in existing_dates]
    partial_dates = [d for d in all_dates if d in existing_dates]

    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    print(f"Dates with data: {sorted(existing_dates & set(all_dates))}")
    print(f"Gap dates (no data): {gap_dates}")
    print(f"Partial dates (have some data): {partial_dates}")

    dates_to_fill = gap_dates
    if not dates_to_fill:
        print("\nNo gaps found!")
        return

    print(f"\nFilling {len(dates_to_fill)} gap dates...")
    total_new = 0

    with ArcticShiftClient() as client:
        for date_str in sorted(dates_to_fill):
            try:
                count = fill_date(client, date_str, storage, existing_ids)
                total_new += count
                print(f"  -> {count} new posts saved for {date_str}")
            except Exception as e:
                print(f"  ERROR on {date_str}: {e}")
                continue

    print(f"\nGap fill complete! {total_new} new posts collected across {len(dates_to_fill)} dates.")


if __name__ == "__main__":
    main()
