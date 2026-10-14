"""
Convert JSONL post files to a single CSV.
Handles both .jsonl and .jsonl.gz files. Streams to keep memory low.

Usage:
  python -m src.jsonl_to_csv --input ~/path/to/historical-data --output ~/path/to/soccer_posts.csv
"""

import argparse
import csv
import gzip
import json
import os
import sys
from datetime import datetime, timezone


COLUMNS = [
    "id", "name", "title", "selftext",
    "author", "author_fullname",
    "author_flair_text", "author_flair_css_class",
    "author_flair_background_color", "author_flair_text_color", "author_flair_type",
    "author_flair_richtext", "author_flair_template_id",
    "author_premium", "author_is_blocked",
    "link_flair_text", "link_flair_css_class",
    "link_flair_background_color", "link_flair_text_color", "link_flair_type",
    "link_flair_richtext", "link_flair_template_id",
    "score", "ups", "downs", "upvote_ratio",
    "num_comments", "num_crossposts",
    "total_awards_received", "gilded",
    "created_utc", "created_datetime",
    "edited",
    "domain", "url", "permalink",
    "is_self", "is_video", "is_original_content", "is_reddit_media_domain",
    "post_hint", "thumbnail", "thumbnail_height", "thumbnail_width",
    "over_18", "spoiler", "stickied", "locked", "pinned",
    "distinguished", "contest_mode",
    "removed_by_category", "banned_by",
    "no_follow", "send_replies",
    "hide_score", "archived", "quarantine",
    "subreddit", "subreddit_id", "subreddit_subscribers", "subreddit_type",
    "media", "media_embed", "secure_media", "secure_media_embed",
    "preview",
    "all_awardings",
    "content_categories", "category",
    "retrieved_on",
    "wls", "pwls",
]


def safe_value(val):
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def post_to_row(post: dict) -> dict:
    created_utc = post.get("created_utc", "")
    created_dt = ""
    if created_utc and isinstance(created_utc, (int, float)):
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    row = {}
    for col in COLUMNS:
        if col == "created_datetime":
            row[col] = created_dt
        else:
            row[col] = safe_value(post.get(col, ""))
    return row


def open_jsonl(filepath):
    if filepath.endswith(".gz"):
        return gzip.open(filepath, "rt", encoding="utf-8")
    return open(filepath, "r", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL post files to CSV")
    parser.add_argument("--input", required=True, help="Directory containing JSONL files")
    parser.add_argument("--output", required=True, help="Output CSV file path")
    args = parser.parse_args()

    input_dir = args.input
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".jsonl") or f.endswith(".jsonl.gz")
        if "posts" in f
    ])

    if not files:
        print("No post JSONL files found in input directory.")
        sys.exit(1)

    print(f"Found {len(files)} post files:")
    for f in files:
        print(f"  {os.path.basename(f)}")

    total = 0
    seen_ids = set()

    with open(args.output, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=COLUMNS)
        writer.writeheader()

        for filepath in files:
            basename = os.path.basename(filepath)
            file_count = 0
            dupes = 0
            print(f"\nProcessing {basename}...")

            with open_jsonl(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        post = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    post_id = post.get("id", "")
                    if post_id in seen_ids:
                        dupes += 1
                        continue
                    seen_ids.add(post_id)

                    writer.writerow(post_to_row(post))
                    file_count += 1
                    total += 1

                    if total % 100000 == 0:
                        print(f"  {total:,} posts written...")

            print(f"  {file_count:,} posts from {basename} ({dupes:,} duplicates skipped)")

    print(f"\nDone! {total:,} unique posts written to {args.output}")
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"File size: {size_mb:,.1f} MB")


if __name__ == "__main__":
    main()
