"""
Historical backfill via Arctic Shift.

Crawls the full r/soccer archive from a start date to an end date, collecting
posts and comments. Designed to run on GitHub Actions in 6-hour chunks, with
each chunk uploading compressed data as a GitHub Release asset.

Resumes from a progress cursor saved in data/index/backfill_progress.json.

Usage:
  python -m src.backfill_arctic_shift                    # resume or start
  python -m src.backfill_arctic_shift --start 2010-01-01 # explicit start
  python -m src.backfill_arctic_shift --chunk-hours 6    # time limit per run
"""

import argparse
import gzip
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.arctic_shift_client import ArcticShiftClient


PROGRESS_FILE = os.path.join("data", "index", "backfill_progress.json")
OUTPUT_DIR = os.path.join("data", "backfill_chunks")


def to_ts(date_str: str) -> int:
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp())


def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress: dict):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_chunk(subreddit: str, start_date: str, end_date: str,
              chunk_hours: float, include_comments: bool):
    start_ts = to_ts(start_date)
    end_ts = to_ts(end_date)

    progress = load_progress()
    chunk_number = progress.get("chunk_number", 0) + 1

    # Resume from where we left off
    posts_cursor = progress.get("posts_cursor_ts")
    comments_cursor = progress.get("comments_cursor_ts")
    posts_done = progress.get("posts_done", False)

    if posts_cursor and not posts_done:
        current_before = posts_cursor
        print(f"Resuming posts from timestamp {current_before}")
    elif not posts_done:
        current_before = end_ts
        print(f"Starting posts from {end_date} back to {start_date}")
    else:
        current_before = None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    deadline = time.time() + chunk_hours * 3600

    posts_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_number:03d}_posts.jsonl.gz")
    comments_file = os.path.join(OUTPUT_DIR, f"chunk_{chunk_number:03d}_comments.jsonl.gz")

    total_posts = progress.get("total_posts", 0)
    total_comments = progress.get("total_comments", 0)
    chunk_posts = 0
    chunk_comments = 0

    with ArcticShiftClient() as client:
        # Phase 1: Posts
        if not posts_done:
            print(f"=== Chunk {chunk_number}: Collecting posts ===")
            with gzip.open(posts_file, "wt", encoding="utf-8") as f:
                while current_before > start_ts:
                    if time.time() > deadline:
                        print(f"Time limit reached. Saving progress at cursor {current_before}")
                        save_progress({
                            "chunk_number": chunk_number,
                            "posts_cursor_ts": current_before,
                            "posts_done": False,
                            "comments_cursor_ts": None,
                            "total_posts": total_posts + chunk_posts,
                            "total_comments": total_comments,
                            "start_date": start_date,
                            "end_date": end_date,
                        })
                        print(f"Chunk {chunk_number}: {chunk_posts} posts written to {posts_file}")
                        return False

                    batch = client._fetch_page("posts/search", subreddit, start_ts, current_before)
                    if not batch:
                        break

                    for item in batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    chunk_posts += len(batch)

                    oldest = min(item["created_utc"] for item in batch)
                    when = datetime.fromtimestamp(oldest, tz=timezone.utc)
                    print(f"  +{len(batch)} posts (total {total_posts + chunk_posts}), "
                          f"at {when:%Y-%m-%d %H:%M}")

                    if oldest >= current_before:
                        break
                    current_before = oldest

            print(f"Posts complete: {total_posts + chunk_posts} total")
            posts_done = True
            total_posts += chunk_posts

        # Phase 2: Comments
        if include_comments:
            if not comments_cursor:
                comments_cursor = end_ts
                print(f"\n=== Chunk {chunk_number}: Collecting comments ===")

            with gzip.open(comments_file, "wt", encoding="utf-8") as f:
                while comments_cursor > start_ts:
                    if time.time() > deadline:
                        print(f"Time limit reached. Saving progress at cursor {comments_cursor}")
                        save_progress({
                            "chunk_number": chunk_number,
                            "posts_cursor_ts": None,
                            "posts_done": True,
                            "comments_cursor_ts": comments_cursor,
                            "total_posts": total_posts,
                            "total_comments": total_comments + chunk_comments,
                            "start_date": start_date,
                            "end_date": end_date,
                        })
                        print(f"Chunk {chunk_number}: {chunk_comments} comments written to {comments_file}")
                        return False

                    batch = client._fetch_page("comments/search", subreddit, start_ts, comments_cursor)
                    if not batch:
                        break

                    for item in batch:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    f.flush()
                    chunk_comments += len(batch)

                    oldest = min(item["created_utc"] for item in batch)
                    when = datetime.fromtimestamp(oldest, tz=timezone.utc)
                    print(f"  +{len(batch)} comments (total {total_comments + chunk_comments}), "
                          f"at {when:%Y-%m-%d %H:%M}")

                    if oldest >= comments_cursor:
                        break
                    comments_cursor = oldest

            total_comments += chunk_comments
            print(f"Comments complete: {total_comments} total")

    # All done
    save_progress({
        "chunk_number": chunk_number,
        "posts_done": True,
        "comments_done": True,
        "total_posts": total_posts,
        "total_comments": total_comments,
        "start_date": start_date,
        "end_date": end_date,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })

    print(f"\n=== Backfill complete ===")
    print(f"Total posts: {total_posts}")
    print(f"Total comments: {total_comments}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Backfill r/soccer from Arctic Shift")
    parser.add_argument("--start", default="2010-01-01",
                        help="Start date (YYYY-MM-DD), default: 2010-01-01")
    parser.add_argument("--end", default="2026-06-08",
                        help="End date (YYYY-MM-DD), default: 2026-06-08")
    parser.add_argument("--chunk-hours", type=float, default=5.5,
                        help="Max hours per chunk (default: 5.5, fits in 6h Actions job)")
    parser.add_argument("--no-comments", action="store_true",
                        help="Skip comment collection")
    args = parser.parse_args()

    progress = load_progress()
    if progress.get("completed_at"):
        print("Backfill already completed. Delete progress file to re-run.")
        print(f"  Completed at: {progress['completed_at']}")
        print(f"  Total posts: {progress['total_posts']}")
        print(f"  Total comments: {progress['total_comments']}")
        return

    start = progress.get("start_date", args.start)
    end = progress.get("end_date", args.end)

    done = run_chunk("soccer", start, end, args.chunk_hours, not args.no_comments)

    if done:
        print("Backfill fully complete!")
    else:
        print("Chunk complete. Run again to continue.")


if __name__ == "__main__":
    main()
