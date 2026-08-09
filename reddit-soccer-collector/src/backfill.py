"""
Backfill script: re-processes all RSS-collected posts using the OAuth API.

Run this ONCE after Reddit API access is approved to:
  1. Re-fetch full post metadata (scores, flair, upvote ratio, etc.)
  2. Re-fetch complete comment trees (up to configured ceilings)
  3. Overwrite RSS-limited data with full API data

Usage:
  python -m src.backfill

Requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD
environment variables (or GitHub Secrets if run via Actions).
"""

import json
import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.logging_config import setup_logging
from src.utils.reddit_client import RedditClient
from src.utils.storage import Storage
from src.utils.media import extract_media_urls, is_reddit_hosted, download_media, verify_url
from src.harvester import collect_comments_from_tree, expand_more_comments, find_more_in_tree


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def backfill_post(client, storage, post, comment_limit, media_config, subreddit, logger):
    post_id = post["id"]
    now = datetime.now(timezone.utc).isoformat()

    logger.info(f"Backfilling post {post_id}: {post['title'][:60]}")

    response = client.get_post_with_comments(subreddit, post_id, limit=500)
    if not response or len(response) < 2:
        logger.error(f"  Failed to fetch post {post_id}")
        return None

    api_post_data = response[0]["data"]["children"][0]["data"]
    comment_listing = response[1]["data"]["children"]

    post["author_fullname"] = api_post_data.get("author_fullname", "")
    post["author_flair_text"] = api_post_data.get("author_flair_text", "")
    post["author_flair_css_class"] = api_post_data.get("author_flair_css_class", "")
    post["link_flair_text"] = api_post_data.get("link_flair_text", "")
    post["link_flair_css_class"] = api_post_data.get("link_flair_css_class", "")
    post["selftext"] = api_post_data.get("selftext", "")
    post["selftext_html"] = api_post_data.get("selftext_html", "")
    post["domain"] = api_post_data.get("domain", "")
    post["is_self"] = api_post_data.get("is_self", False)
    post["is_video"] = api_post_data.get("is_video", False)
    post["is_original_content"] = api_post_data.get("is_original_content", False)
    post["over_18"] = api_post_data.get("over_18", False)
    post["spoiler"] = api_post_data.get("spoiler", False)
    post["stickied"] = api_post_data.get("stickied", False)
    post["distinguished"] = api_post_data.get("distinguished")
    post["num_crossposts"] = api_post_data.get("num_crossposts", 0)
    post["crosspost_parent"] = api_post_data.get("crosspost_parent")
    post["total_awards_received"] = api_post_data.get("total_awards_received", 0)
    post["all_awardings"] = api_post_data.get("all_awardings", [])
    post["thumbnail"] = api_post_data.get("thumbnail", "")
    post["url"] = api_post_data.get("url", post.get("url", ""))

    backfill_snapshot = {
        "collected_at": now,
        "score": api_post_data.get("score", 0),
        "upvote_ratio": api_post_data.get("upvote_ratio", 0.0),
        "num_comments": api_post_data.get("num_comments", 0),
        "author_flair_text": api_post_data.get("author_flair_text", ""),
        "link_flair_text": api_post_data.get("link_flair_text", ""),
        "edited": api_post_data.get("edited", False),
        "deleted": api_post_data.get("author") == "[deleted]",
        "removed": api_post_data.get("selftext") == "[removed]",
        "selftext": api_post_data.get("selftext", ""),
    }

    if post.get("t48_snapshot") and post["t48_snapshot"].get("note"):
        post["t48_snapshot"] = backfill_snapshot
    else:
        post["backfill_snapshot"] = backfill_snapshot

    media_urls = extract_media_urls(api_post_data)
    for media_entry in media_urls:
        url = media_entry["url"]
        if media_config["download_reddit_hosted"] and is_reddit_hosted(url):
            existing = [m for m in post.get("media_urls", []) if m.get("url") == url and m.get("downloaded")]
            if not existing:
                save_dir = os.path.join(storage.media_dir, post_id)
                result = download_media(
                    url, save_dir,
                    timeout=media_config["download_timeout_seconds"],
                    max_size_mb=media_config["max_file_size_mb"],
                )
                media_entry.update(result)
        elif media_config["verify_external_links"] and media_entry["source"] == "external_link":
            result = verify_url(url)
            media_entry.update(result)
    post["media_urls"] = media_urls

    comments = collect_comments_from_tree(comment_listing)
    logger.info(f"  Initial comment tree: {len(comments)} comments")

    more_ids = []
    for child in comment_listing:
        if child.get("kind") == "more":
            more_ids.extend(child.get("data", {}).get("children", []))
    more_ids.extend(find_more_in_tree(comment_listing))

    if more_ids and len(comments) < comment_limit:
        logger.info(f"  Expanding {len(more_ids)} 'more' comment stubs")
        link_fullname = f"t3_{post_id}"
        expanded = expand_more_comments(
            client, link_fullname, more_ids,
            comment_limit, len(comments), logger,
        )
        comments.extend(expanded)
        logger.info(f"  After expansion: {len(comments)} comments")

    was_truncated = len(comments) >= comment_limit
    if was_truncated:
        comments = comments[:comment_limit]
        storage.log_truncated(post_id, comment_limit, len(comments))
        logger.warning(f"  Post {post_id} truncated at {comment_limit} comments")

    post["comments"] = comments
    post["collection_metadata"] = {
        "total_comments_collected": len(comments),
        "comment_limit_applied": comment_limit,
        "was_truncated": was_truncated,
        "harvested_at": now,
        "backfilled": True,
    }
    post["source"] = "backfilled_from_rss"

    return post


def run():
    config = load_config()
    logger = setup_logging("backfill", config["storage"]["log_dir"])

    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    username = os.environ.get("REDDIT_USERNAME", "")
    password = os.environ.get("REDDIT_PASSWORD", "")

    if not all([client_id, client_secret, username, password]):
        logger.error(
            "OAuth credentials required. Set environment variables: "
            "REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD"
        )
        print("\nBackfill requires API credentials. Set these environment variables:")
        print("  export REDDIT_CLIENT_ID='your_client_id'")
        print("  export REDDIT_CLIENT_SECRET='your_client_secret'")
        print("  export REDDIT_USERNAME='your_username'")
        print("  export REDDIT_PASSWORD='your_password'")
        return

    client = RedditClient(
        user_agent=config["reddit"]["user_agent"],
        requests_per_minute=config["reddit"]["requests_per_minute"],
        mode="oauth",
        client_id=client_id,
        client_secret=client_secret,
        username=username,
        password=password,
    )

    storage = Storage(config["storage"]["data_dir"])
    subreddit = config["subreddit"]
    media_config = config["media"]
    default_limit = config["comment_limits"]["default"]
    match_limit = config["comment_limits"]["match_thread"]

    logger.info("Starting backfill of RSS-collected posts")

    rss_posts = []
    collected_dir = storage.collected_dir
    if os.path.exists(collected_dir):
        for batch_name in sorted(os.listdir(collected_dir)):
            batch_dir = os.path.join(collected_dir, batch_name)
            if not os.path.isdir(batch_dir):
                continue
            for filename in sorted(os.listdir(batch_dir)):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(batch_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    post = json.load(f)
                if post.get("source") in ("rss", "backfilled_from_rss"):
                    if post.get("source") == "backfilled_from_rss":
                        continue
                    rss_posts.append((batch_name, post))

    if not rss_posts:
        logger.info("No RSS-collected posts found to backfill.")
        print("No RSS-collected posts found. Nothing to backfill.")
        return

    logger.info(f"Found {len(rss_posts)} RSS-collected posts to backfill")
    print(f"Found {len(rss_posts)} posts to backfill. This may take a while...")

    processed = 0
    failed = 0
    total_comments = 0

    for batch_name, post in rss_posts:
        post_id = post["id"]
        comment_limit = match_limit if post.get("is_match_thread") else default_limit

        try:
            result = backfill_post(
                client, storage, post, comment_limit, media_config, subreddit, logger,
            )
            if result:
                storage.save_collected_post(batch_name, post_id, result)
                total_comments += len(result.get("comments", []))
                processed += 1
                if processed % 10 == 0:
                    print(f"  Progress: {processed}/{len(rss_posts)} posts backfilled")
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Error backfilling post {post_id}: {e}")
            failed += 1

    logger.info(
        f"Backfill complete. {processed} posts backfilled, {failed} failed, "
        f"{total_comments} total comments collected."
    )
    print(f"\nBackfill complete:")
    print(f"  Posts backfilled: {processed}")
    print(f"  Failed: {failed}")
    print(f"  Total comments collected: {total_comments}")


if __name__ == "__main__":
    run()
