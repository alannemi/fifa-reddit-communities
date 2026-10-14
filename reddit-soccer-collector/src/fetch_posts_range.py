"""
Fetch all r/soccer posts from Arctic Shift for a given date range.
Outputs a single compressed JSONL file.

Usage:
  python -m src.fetch_posts_range --start 2026-06-09 --end 2026-08-01 --output ~/path/to/output.jsonl.gz
"""

import argparse
import gzip
import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.arctic_shift_client import ArcticShiftClient


def main():
    parser = argparse.ArgumentParser(description="Fetch r/soccer posts from Arctic Shift")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date exclusive (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output file path (.jsonl or .jsonl.gz)")
    args = parser.parse_args()

    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    after_ts = int(start_dt.timestamp())
    before_ts = int(end_dt.timestamp())

    print(f"Fetching r/soccer posts from {args.start} to {args.end}...")
    print(f"Output: {args.output}")

    total = 0
    open_fn = gzip.open if args.output.endswith(".gz") else open

    with ArcticShiftClient() as client:
        cursor = before_ts
        with open_fn(args.output, "wt", encoding="utf-8") as f:
            while cursor > after_ts:
                batch = client._fetch_page("posts/search", "soccer", after_ts, cursor)
                if not batch:
                    break

                for post in batch:
                    f.write(json.dumps(post, ensure_ascii=False) + "\n")
                    total += 1

                oldest = min(item["created_utc"] for item in batch)
                oldest_date = datetime.fromtimestamp(oldest, tz=timezone.utc).strftime("%Y-%m-%d")
                print(f"  {total:,} posts collected (cursor at {oldest_date})")

                if oldest >= cursor:
                    break
                cursor = oldest

    print(f"\nDone! {total:,} posts saved to {args.output}")


if __name__ == "__main__":
    main()
