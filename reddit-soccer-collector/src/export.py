"""
Export pipeline: converts collected JSON files into CSV files for R analysis.
Produces three files:
  - posts.csv: one row per post with t0 and t48 snapshots
  - comments.csv: one row per comment with post_id foreign key
  - collection_summary.csv: one row per batch with collection stats

Handles both Arctic Shift (full schema) and RSS (limited schema) data.
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.storage import Storage


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


POST_COLUMNS = [
    "post_id", "title", "selftext", "selftext_t48",
    "author", "author_fullname",
    "author_flair_text_t0", "author_flair_text_t48",
    "author_flair_richtext",
    "link_flair_text_t0", "link_flair_text_t48",
    "link_flair_richtext", "link_flair_background_color",
    "score_t0", "score_t48", "score_delta",
    "upvote_ratio_t0", "upvote_ratio_t48",
    "ups", "downs",
    "num_comments_t0", "num_comments_t48", "num_comments_delta",
    "created_utc", "created_datetime",
    "edited_t0", "edited_t48",
    "deleted_t48", "removed_t48",
    "removed_by_category",
    "url", "permalink", "domain",
    "is_self", "is_video", "is_original_content",
    "over_18", "spoiler", "stickied", "distinguished",
    "is_match_thread",
    "num_crossposts", "total_awards_received",
    "thumbnail", "post_hint",
    "media_urls",
    "source",
    "t0_collected_at", "t48_collected_at", "batch_id",
    "comments_collected", "comment_limit_applied", "was_truncated",
]

COMMENT_COLUMNS = [
    "comment_id", "post_id",
    "author", "author_fullname",
    "author_flair_text",
    "author_flair_richtext",
    "body", "score", "ups", "downs",
    "created_utc", "created_datetime",
    "edited", "parent_id", "link_id",
    "is_submitter", "distinguished", "stickied",
    "controversiality", "collapsed", "collapsed_reason_code",
    "total_awards_received",
    "source",
]


def safe_json(val):
    """Serialize complex values to JSON string for CSV."""
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return val


def post_to_row(post: dict) -> dict:
    t0 = post.get("t0_snapshot") or {}
    t48 = post.get("t48_snapshot") or {}
    meta = post.get("collection_metadata") or {}

    score_t0 = t0.get("score", post.get("score", 0))
    score_t48 = t48.get("score")
    num_t0 = t0.get("num_comments", post.get("num_comments", 0))
    num_t48 = t48.get("num_comments")

    created_utc = post.get("created_utc", "")
    created_dt = ""
    if created_utc and isinstance(created_utc, (int, float)):
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

    return {
        "post_id": post.get("id", ""),
        "title": post.get("title", ""),
        "selftext": post.get("selftext", ""),
        "selftext_t48": t48.get("selftext", ""),
        "author": post.get("author", ""),
        "author_fullname": post.get("author_fullname", ""),
        "author_flair_text_t0": t0.get("author_flair_text", post.get("author_flair_text", "")),
        "author_flair_text_t48": t48.get("author_flair_text", ""),
        "author_flair_richtext": safe_json(post.get("author_flair_richtext", "")),
        "link_flair_text_t0": t0.get("link_flair_text", post.get("link_flair_text", "")),
        "link_flair_text_t48": t48.get("link_flair_text", ""),
        "link_flair_richtext": safe_json(post.get("link_flair_richtext", "")),
        "link_flair_background_color": post.get("link_flair_background_color", ""),
        "score_t0": score_t0,
        "score_t48": score_t48 if score_t48 is not None else "",
        "score_delta": (score_t48 - score_t0) if score_t48 is not None else "",
        "upvote_ratio_t0": t0.get("upvote_ratio", post.get("upvote_ratio", "")),
        "upvote_ratio_t48": t48.get("upvote_ratio", ""),
        "ups": post.get("ups", ""),
        "downs": post.get("downs", ""),
        "num_comments_t0": num_t0,
        "num_comments_t48": num_t48 if num_t48 is not None else "",
        "num_comments_delta": (num_t48 - num_t0) if num_t48 is not None else "",
        "created_utc": created_utc,
        "created_datetime": created_dt or post.get("created_datetime", ""),
        "edited_t0": t0.get("edited", post.get("edited", "")),
        "edited_t48": t48.get("edited", ""),
        "deleted_t48": t48.get("deleted", ""),
        "removed_t48": t48.get("removed", ""),
        "removed_by_category": post.get("removed_by_category", ""),
        "url": post.get("url", ""),
        "permalink": post.get("permalink", ""),
        "domain": post.get("domain", ""),
        "is_self": post.get("is_self", ""),
        "is_video": post.get("is_video", ""),
        "is_original_content": post.get("is_original_content", ""),
        "over_18": post.get("over_18", ""),
        "spoiler": post.get("spoiler", ""),
        "stickied": post.get("stickied", ""),
        "distinguished": post.get("distinguished", ""),
        "is_match_thread": post.get("is_match_thread", ""),
        "num_crossposts": post.get("num_crossposts", ""),
        "total_awards_received": post.get("total_awards_received", ""),
        "thumbnail": post.get("thumbnail", ""),
        "post_hint": post.get("post_hint", ""),
        "media_urls": safe_json(post.get("media_urls", [])),
        "source": post.get("source", ""),
        "t0_collected_at": t0.get("collected_at", ""),
        "t48_collected_at": t48.get("collected_at", ""),
        "batch_id": t0.get("batch_id", ""),
        "comments_collected": meta.get("total_comments_collected", ""),
        "comment_limit_applied": meta.get("comment_limit_applied", ""),
        "was_truncated": meta.get("was_truncated", ""),
    }


def comment_to_row(comment: dict, post_id: str) -> dict:
    created_utc = comment.get("created_utc", "")
    created_dt = ""
    if created_utc and isinstance(created_utc, (int, float)):
        created_dt = datetime.fromtimestamp(created_utc, tz=timezone.utc).isoformat()

    return {
        "comment_id": comment.get("id", ""),
        "post_id": post_id,
        "author": comment.get("author", ""),
        "author_fullname": comment.get("author_fullname", ""),
        "author_flair_text": comment.get("author_flair_text", ""),
        "author_flair_richtext": safe_json(comment.get("author_flair_richtext", "")),
        "body": comment.get("body", ""),
        "score": comment.get("score", ""),
        "ups": comment.get("ups", ""),
        "downs": comment.get("downs", ""),
        "created_utc": created_utc,
        "created_datetime": created_dt or comment.get("created_datetime", ""),
        "edited": comment.get("edited", ""),
        "parent_id": comment.get("parent_id", ""),
        "link_id": comment.get("link_id", ""),
        "is_submitter": comment.get("is_submitter", ""),
        "distinguished": comment.get("distinguished", ""),
        "stickied": comment.get("stickied", ""),
        "controversiality": comment.get("controversiality", ""),
        "collapsed": comment.get("collapsed", ""),
        "collapsed_reason_code": comment.get("collapsed_reason_code", ""),
        "total_awards_received": comment.get("total_awards_received", ""),
        "source": comment.get("source", ""),
    }


def run():
    config = load_config()
    storage = Storage(config["storage"]["data_dir"])

    print("Loading collected posts...")
    posts = storage.get_all_collected()
    print(f"Found {len(posts)} posts")

    if not posts:
        print("No collected posts found. Nothing to export.")
        return

    export_dir = storage.export_dir
    os.makedirs(export_dir, exist_ok=True)

    posts_csv = os.path.join(export_dir, "posts.csv")
    comments_csv = os.path.join(export_dir, "comments.csv")
    summary_csv = os.path.join(export_dir, "collection_summary.csv")

    total_comments = 0
    batch_stats = {}

    print("Writing posts.csv...")
    with open(posts_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=POST_COLUMNS)
        writer.writeheader()
        for post in posts:
            writer.writerow(post_to_row(post))
            batch_id = (post.get("t0_snapshot") or {}).get("batch_id", "unknown")
            if batch_id not in batch_stats:
                batch_stats[batch_id] = {"posts": 0, "comments": 0, "truncated": 0}
            batch_stats[batch_id]["posts"] += 1
            meta = post.get("collection_metadata") or {}
            n_comments = meta.get("total_comments_collected", 0) or 0
            batch_stats[batch_id]["comments"] += n_comments
            if meta.get("was_truncated"):
                batch_stats[batch_id]["truncated"] += 1

    print("Writing comments.csv...")
    with open(comments_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMMENT_COLUMNS)
        writer.writeheader()
        for post in posts:
            post_id = post.get("id", "")
            for comment in post.get("comments", []):
                writer.writerow(comment_to_row(comment, post_id))
                total_comments += 1

    print("Writing collection_summary.csv...")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["batch_id", "posts_collected", "comments_collected", "truncated_posts"])
        writer.writeheader()
        for batch_id in sorted(batch_stats.keys()):
            stats = batch_stats[batch_id]
            writer.writerow({
                "batch_id": batch_id,
                "posts_collected": stats["posts"],
                "comments_collected": stats["comments"],
                "truncated_posts": stats["truncated"],
            })

    print(f"\nExport complete:")
    print(f"  Posts:    {len(posts)} -> {posts_csv}")
    print(f"  Comments: {total_comments} -> {comments_csv}")
    print(f"  Summary:  {len(batch_stats)} batches -> {summary_csv}")


if __name__ == "__main__":
    run()
