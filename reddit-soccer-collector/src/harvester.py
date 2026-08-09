"""
Agent 2: Harvester
Runs every 5 minutes. Processes posts from 48 hours ago — re-fetches post metadata
and collects the full comment tree.

Supports three modes:
  - "arctic_shift+rss": Dual-source — Arctic Shift primary, RSS failsafe (default)
  - "rss": Uses Reddit's RSS feed for comments (limited: no scores, no threading)
  - "oauth": Uses Reddit's authenticated API (full metadata, full comment tree)
"""

import os
import sys
import time
from datetime import datetime, timezone, timedelta

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.logging_config import setup_logging
from src.utils.reddit_client import RedditClient
from src.utils.arctic_shift_client import ArcticShiftClient
from src.utils.storage import Storage


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_client(config: dict) -> RedditClient:
    reddit_config = config["reddit"]
    mode = config.get("mode", "rss")

    if mode == "oauth":
        return RedditClient(
            user_agent=reddit_config["user_agent"],
            requests_per_minute=reddit_config["requests_per_minute"],
            mode="oauth",
            client_id=os.environ.get("REDDIT_CLIENT_ID", ""),
            client_secret=os.environ.get("REDDIT_CLIENT_SECRET", ""),
            username=os.environ.get("REDDIT_USERNAME", ""),
            password=os.environ.get("REDDIT_PASSWORD", ""),
        )
    else:
        return RedditClient(
            user_agent=reddit_config["user_agent"],
            requests_per_minute=reddit_config["requests_per_minute"],
            mode="rss",
        )


def extract_comment(comment_data: dict) -> dict:
    if comment_data.get("kind") != "t1":
        return None

    data = comment_data.get("data", {})
    if not data.get("id"):
        return None

    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "author": data.get("author", "[deleted]"),
        "author_fullname": data.get("author_fullname", ""),
        "author_flair_text": data.get("author_flair_text", ""),
        "author_flair_css_class": data.get("author_flair_css_class", ""),
        "body": data.get("body", ""),
        "body_html": data.get("body_html", ""),
        "score": data.get("score", 0),
        "ups": data.get("ups", 0),
        "downs": data.get("downs", 0),
        "created_utc": data.get("created_utc"),
        "created_datetime": datetime.fromtimestamp(
            data.get("created_utc", 0), tz=timezone.utc
        ).isoformat() if data.get("created_utc") else None,
        "edited": data.get("edited", False),
        "parent_id": data.get("parent_id", ""),
        "depth": data.get("depth", 0),
        "is_submitter": data.get("is_submitter", False),
        "distinguished": data.get("distinguished"),
        "stickied": data.get("stickied", False),
        "controversiality": data.get("controversiality", 0),
        "total_awards_received": data.get("total_awards_received", 0),
        "all_awardings": data.get("all_awardings", []),
        "source": "oauth",
    }


def collect_comments_from_tree(children: list) -> list:
    comments = []
    for child in children:
        comment = extract_comment(child)
        if comment:
            comments.append(comment)
        data = child.get("data", {})
        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            comments.extend(collect_comments_from_tree(reply_children))
    return comments


def expand_more_comments(client: RedditClient, link_fullname: str, more_children_ids: list,
                         comment_limit: int, current_count: int, logger) -> list:
    all_comments = []
    batch_size = 100

    for i in range(0, len(more_children_ids), batch_size):
        if current_count + len(all_comments) >= comment_limit:
            logger.info(f"Hit comment limit ({comment_limit}) during expansion")
            break

        batch = more_children_ids[i:i + batch_size]
        try:
            response = client.get_more_comments(link_fullname, batch)
            if not response:
                continue

            things = response.get("json", {}).get("data", {}).get("things", [])
            for thing in things:
                comment = extract_comment(thing)
                if comment:
                    all_comments.append(comment)
        except Exception as e:
            logger.warning(f"Error expanding more comments: {e}")
            continue

    return all_comments


def find_more_in_tree(children):
    ids = []
    for child in children:
        if child.get("kind") == "more":
            ids.extend(child.get("data", {}).get("children", []))
        data = child.get("data", {})
        replies = data.get("replies")
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            ids.extend(find_more_in_tree(reply_children))
    return ids


def process_post_arctic_shift(as_client: ArcticShiftClient, storage: Storage,
                              post: dict, comment_limit: int, config: dict,
                              logger) -> dict:
    """Process a post using Arctic Shift: re-fetch metadata + collect comments."""
    post_id = post["id"]
    subreddit = config["subreddit"]
    now = datetime.now(timezone.utc).isoformat()

    logger.info(f"Processing post {post_id} (Arctic Shift): {post['title'][:60]}")

    # Re-fetch post for t48 snapshot
    updated_post = as_client.get_post_by_id(subreddit, post_id)
    if updated_post:
        post["t48_snapshot"] = {
            "collected_at": now,
            "score": updated_post.get("score", 0),
            "upvote_ratio": updated_post.get("upvote_ratio", 0.0),
            "num_comments": updated_post.get("num_comments", 0),
            "author_flair_text": updated_post.get("author_flair_text", ""),
            "link_flair_text": updated_post.get("link_flair_text", ""),
            "edited": updated_post.get("edited", False),
            "deleted": updated_post.get("author") == "[deleted]",
            "removed": updated_post.get("selftext") == "[removed]",
            "selftext": updated_post.get("selftext", ""),
        }
    else:
        logger.warning(f"  Could not re-fetch post {post_id} from Arctic Shift")
        post["t48_snapshot"] = {
            "collected_at": now,
            "note": "Arctic Shift post re-fetch failed",
        }

    # Fetch comments
    comments_raw = as_client.get_comments_for_post(subreddit, post_id)
    logger.info(f"  Arctic Shift returned {len(comments_raw)} comments")

    # Arctic Shift comments come as flat records with full metadata
    comments = []
    for c in comments_raw:
        if not c.get("id"):
            continue
        comments.append(dict(c))  # preserve full Arctic Shift comment schema

    # Enforce comment ceiling
    was_truncated = len(comments) > comment_limit
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
        "comment_source": "arctic_shift",
    }

    return post


def process_post_arctic_shift_with_rss_fallback(as_client: ArcticShiftClient,
                                                 rss_client: RedditClient,
                                                 storage: Storage, post: dict,
                                                 comment_limit: int, config: dict,
                                                 logger) -> dict:
    """Try Arctic Shift first, fall back to RSS for comments if AS fails."""
    try:
        return process_post_arctic_shift(as_client, storage, post, comment_limit, config, logger)
    except Exception as e:
        logger.warning(f"  Arctic Shift failed for post {post['id']}: {e}. Trying RSS.")
        return process_post_rss(rss_client, storage, post, config, logger)


def process_post_oauth(client: RedditClient, storage: Storage, post: dict,
                       comment_limit: int, config: dict, logger) -> dict:
    post_id = post["id"]
    subreddit = config["subreddit"]
    now = datetime.now(timezone.utc).isoformat()

    logger.info(f"Processing post {post_id}: {post['title'][:60]}")

    response = client.get_post_with_comments(subreddit, post_id, limit=500)
    if not response or len(response) < 2:
        logger.error(f"Failed to fetch post {post_id}")
        return None

    updated_post_data = response[0]["data"]["children"][0]["data"]
    comment_listing = response[1]["data"]["children"]

    post["t48_snapshot"] = {
        "collected_at": now,
        "score": updated_post_data.get("score", 0),
        "upvote_ratio": updated_post_data.get("upvote_ratio", 0.0),
        "num_comments": updated_post_data.get("num_comments", 0),
        "author_flair_text": updated_post_data.get("author_flair_text", ""),
        "link_flair_text": updated_post_data.get("link_flair_text", ""),
        "edited": updated_post_data.get("edited", False),
        "deleted": updated_post_data.get("author") == "[deleted]",
        "removed": updated_post_data.get("selftext") == "[removed]",
        "selftext": updated_post_data.get("selftext", ""),
    }

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
    }

    return post


def process_post_rss(client: RedditClient, storage: Storage, post: dict,
                     config: dict, logger) -> dict:
    post_id = post["id"]
    subreddit = config["subreddit"]
    now = datetime.now(timezone.utc).isoformat()

    logger.info(f"Processing post {post_id} (RSS): {post['title'][:60]}")

    result = client.get_post_rss(subreddit, post_id)
    if not result:
        logger.error(f"Failed to fetch comments for post {post_id}")
        return None

    comments = result.get("comments", [])
    logger.info(f"  Collected {len(comments)} comments via RSS")

    post["t48_snapshot"] = {
        "collected_at": now,
        "note": "RSS mode — scores/flair/deletion status unavailable",
    }

    post["comments"] = comments
    post["collection_metadata"] = {
        "total_comments_collected": len(comments),
        "comment_limit_applied": "rss_max",
        "was_truncated": False,
        "harvested_at": now,
        "note": "RSS mode — limited to ~500 comments, no scores/flair/parent chain",
    }

    return post


def get_pending_batches_for_target(storage: Storage, target_time: datetime,
                                   interval_minutes: int) -> list:
    """Find all pending batch IDs that fall within the target time window.
    With 5-minute batches, we need to check all batches from the target hour."""
    target_date = target_time.strftime("%Y-%m-%dT%H")
    all_posts = []

    if not os.path.exists(storage.pending_dir):
        return all_posts

    for batch_name in sorted(os.listdir(storage.pending_dir)):
        if not os.path.isdir(os.path.join(storage.pending_dir, batch_name)):
            continue
        # Match batches that start with the target hour, or the exact minute batch
        if batch_name.startswith(target_date):
            batch_posts = storage.get_pending_batch(batch_name)
            for post in batch_posts:
                post["_batch_id"] = batch_name
            all_posts.extend(batch_posts)

    return all_posts


def run():
    config = load_config()
    logger = setup_logging("harvester", config["storage"]["log_dir"])

    now = datetime.now(timezone.utc)
    end_date = datetime.fromisoformat(config["collection_end_date"])
    harvester_end = end_date + timedelta(hours=config["comment_delay_hours"])
    if now > harvester_end:
        logger.info("Harvester end date has passed (collection end + 48h). Exiting.")
        return

    delay_hours = config["comment_delay_hours"]
    target_time = now - timedelta(hours=delay_hours)
    interval_minutes = config.get("polling_interval_minutes", 5)

    mode = config.get("mode", "rss")
    logger.info(f"Harvester starting. Target time: {target_time.isoformat()}. Mode: {mode}")

    storage = Storage(config["storage"]["data_dir"])

    # Collect pending posts from all batches in the target hour
    pending_posts = get_pending_batches_for_target(storage, target_time, interval_minutes)
    if not pending_posts:
        logger.info(f"No pending posts for target time {target_time.strftime('%Y-%m-%dT%H')}. Nothing to do.")
        return

    logger.info(f"Found {len(pending_posts)} posts to process")

    default_limit = config["comment_limits"]["default"]
    match_limit = config["comment_limits"]["match_thread"]

    processed = 0
    failed = 0
    total_comments = 0

    as_client = None
    rss_client = None

    if mode == "arctic_shift+rss":
        as_config = config.get("arctic_shift", {})
        rate_limit = as_config.get("rate_limit_seconds", 1.0)
        as_client = ArcticShiftClient(rate_limit=rate_limit)
        rss_client = create_client({**config, "mode": "rss"})
    elif mode == "oauth":
        oauth_client = create_client(config)
    else:
        rss_client = create_client(config)

    try:
        for post in pending_posts:
            post_id = post["id"]
            batch_id = post.pop("_batch_id", "unknown")

            try:
                if mode == "arctic_shift+rss":
                    comment_limit = match_limit if post.get("is_match_thread") else default_limit
                    result = process_post_arctic_shift_with_rss_fallback(
                        as_client, rss_client, storage, post, comment_limit, config, logger
                    )
                elif mode == "oauth":
                    comment_limit = match_limit if post.get("is_match_thread") else default_limit
                    result = process_post_oauth(oauth_client, storage, post, comment_limit, config, logger)
                else:
                    result = process_post_rss(rss_client, storage, post, config, logger)

                if result:
                    storage.save_collected_post(batch_id, post_id, result)
                    storage.remove_pending_post(batch_id, post_id)
                    total_comments += len(result.get("comments", []))
                    processed += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error processing post {post_id}: {e}")
                failed += 1
    finally:
        if as_client:
            as_client.close()

    logger.info(
        f"Harvester complete ({mode}). {processed} posts processed, {failed} failed, "
        f"{total_comments} total comments collected."
    )


if __name__ == "__main__":
    run()
