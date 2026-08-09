"""
Agent 1: Scanner
Runs every 5 minutes. Collects new posts from r/soccer with full metadata and media.
Does not collect comments (that's the Harvester's job 48 hours later).

Supports three modes:
  - "arctic_shift+rss": Dual-source — Arctic Shift primary, RSS failsafe (default)
  - "rss": Uses Reddit's public RSS feeds (no auth required, limited metadata)
  - "oauth": Uses Reddit's authenticated API (full metadata, requires credentials)
"""

import os
import sys
from datetime import datetime, timezone

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.logging_config import setup_logging
from src.utils.reddit_client import RedditClient
from src.utils.arctic_shift_client import ArcticShiftClient
from src.utils.storage import Storage
from src.utils.media import extract_media_urls, is_reddit_hosted, download_media, verify_url


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_match_thread(post_data: dict, patterns: list) -> bool:
    title = (post_data.get("title") or "").strip()
    flair = post_data.get("link_flair_text") or ""
    for pattern in patterns:
        if pattern.lower() in title.lower() or pattern.lower() in flair.lower():
            return True
    return False


def extract_post_arctic_shift(post_data: dict, batch_id: str, match_thread_patterns: list) -> dict:
    """Build a collected post record from an Arctic Shift post object.
    Preserves the full Arctic Shift schema and adds collection metadata."""
    now = datetime.now(timezone.utc).isoformat()

    post = dict(post_data)

    post["is_match_thread"] = is_match_thread(post_data, match_thread_patterns)
    post["media_urls"] = []
    post["source"] = "arctic_shift"
    post["t0_snapshot"] = {
        "collected_at": now,
        "batch_id": batch_id,
        "score": post_data.get("score", 0),
        "upvote_ratio": post_data.get("upvote_ratio", 0.0),
        "num_comments": post_data.get("num_comments", 0),
        "author_flair_text": post_data.get("author_flair_text", ""),
        "link_flair_text": post_data.get("link_flair_text", ""),
        "edited": post_data.get("edited", False),
    }
    post["t48_snapshot"] = None
    post["comments"] = []
    post["collection_metadata"] = None

    return post


def extract_post_fields_oauth(post_data: dict, batch_id: str, match_thread_patterns: list) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": post_data.get("id"),
        "name": post_data.get("name"),
        "title": post_data.get("title"),
        "selftext": post_data.get("selftext", ""),
        "selftext_html": post_data.get("selftext_html", ""),
        "author": post_data.get("author", "[deleted]"),
        "author_fullname": post_data.get("author_fullname", ""),
        "author_flair_text": post_data.get("author_flair_text", ""),
        "author_flair_css_class": post_data.get("author_flair_css_class", ""),
        "link_flair_text": post_data.get("link_flair_text", ""),
        "link_flair_css_class": post_data.get("link_flair_css_class", ""),
        "score": post_data.get("score", 0),
        "upvote_ratio": post_data.get("upvote_ratio", 0.0),
        "ups": post_data.get("ups", 0),
        "downs": post_data.get("downs", 0),
        "num_comments": post_data.get("num_comments", 0),
        "created_utc": post_data.get("created_utc"),
        "created_datetime": datetime.fromtimestamp(
            post_data.get("created_utc", 0), tz=timezone.utc
        ).isoformat() if post_data.get("created_utc") else None,
        "edited": post_data.get("edited", False),
        "url": post_data.get("url", ""),
        "permalink": post_data.get("permalink", ""),
        "domain": post_data.get("domain", ""),
        "is_self": post_data.get("is_self", False),
        "is_video": post_data.get("is_video", False),
        "is_original_content": post_data.get("is_original_content", False),
        "over_18": post_data.get("over_18", False),
        "spoiler": post_data.get("spoiler", False),
        "stickied": post_data.get("stickied", False),
        "distinguished": post_data.get("distinguished"),
        "num_crossposts": post_data.get("num_crossposts", 0),
        "crosspost_parent": post_data.get("crosspost_parent"),
        "total_awards_received": post_data.get("total_awards_received", 0),
        "all_awardings": post_data.get("all_awardings", []),
        "thumbnail": post_data.get("thumbnail", ""),
        "is_match_thread": is_match_thread(post_data, match_thread_patterns),
        "media_urls": [],
        "source": "oauth",
        "t0_snapshot": {
            "collected_at": now,
            "batch_id": batch_id,
            "score": post_data.get("score", 0),
            "upvote_ratio": post_data.get("upvote_ratio", 0.0),
            "num_comments": post_data.get("num_comments", 0),
            "author_flair_text": post_data.get("author_flair_text", ""),
            "link_flair_text": post_data.get("link_flair_text", ""),
            "edited": post_data.get("edited", False),
        },
        "t48_snapshot": None,
        "comments": [],
        "collection_metadata": None,
    }


def extract_post_fields_rss(post_data: dict, batch_id: str, match_thread_patterns: list) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    return {
        "id": post_data.get("id"),
        "name": post_data.get("name"),
        "title": post_data.get("title", ""),
        "selftext": post_data.get("selftext", ""),
        "selftext_html": post_data.get("selftext_html", ""),
        "content_html_raw": post_data.get("content_html_raw", ""),
        "author": post_data.get("author", "[unknown]"),
        "author_fullname": "",
        "author_flair_text": "",
        "author_flair_css_class": "",
        "link_flair_text": "",
        "link_flair_css_class": "",
        "score": 0,
        "upvote_ratio": 0.0,
        "ups": 0,
        "downs": 0,
        "num_comments": 0,
        "created_utc": post_data.get("created_utc"),
        "created_datetime": post_data.get("created_datetime"),
        "edited": False,
        "url": post_data.get("url", ""),
        "permalink": post_data.get("permalink", ""),
        "domain": "",
        "is_self": False,
        "is_video": False,
        "is_original_content": False,
        "over_18": False,
        "spoiler": False,
        "stickied": False,
        "distinguished": None,
        "num_crossposts": 0,
        "crosspost_parent": None,
        "total_awards_received": 0,
        "all_awardings": [],
        "thumbnail": "",
        "is_match_thread": is_match_thread(post_data, match_thread_patterns),
        "media_urls": [],
        "source": "rss",
        "t0_snapshot": {
            "collected_at": now,
            "batch_id": batch_id,
            "score": 0,
            "upvote_ratio": 0.0,
            "num_comments": 0,
            "author_flair_text": "",
            "link_flair_text": "",
            "edited": False,
            "note": "RSS mode — scores/flair unavailable at scan time",
        },
        "t48_snapshot": None,
        "comments": [],
        "collection_metadata": None,
    }


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


def merge_arctic_shift_over_rss(as_post: dict, rss_post: dict) -> dict:
    """Merge Arctic Shift data over an RSS post. AS fields take priority."""
    merged = dict(as_post)
    merged["source"] = "merged"
    merged["rss_content_html_raw"] = rss_post.get("content_html_raw", "")
    return merged


def process_media(post: dict, post_data_for_media: dict, storage: Storage,
                  media_config: dict, post_id: str) -> tuple:
    """Extract, download, and verify media. Returns (media_urls, downloaded_count, verified_count)."""
    media_downloaded = 0
    media_verified = 0

    media_urls = extract_media_urls(post_data_for_media)
    for media_entry in media_urls:
        url = media_entry["url"]
        if media_config["download_reddit_hosted"] and is_reddit_hosted(url):
            save_dir = os.path.join(storage.media_dir, post_id)
            result = download_media(
                url, save_dir,
                timeout=media_config["download_timeout_seconds"],
                max_size_mb=media_config["max_file_size_mb"],
            )
            media_entry.update(result)
            if result.get("downloaded"):
                media_downloaded += 1
        elif media_config["verify_external_links"] and media_entry["source"] == "external_link":
            result = verify_url(url)
            media_entry.update(result)
            media_verified += 1

    return media_urls, media_downloaded, media_verified


def run_arctic_shift_rss(config, storage, logger, batch_id):
    """Dual-source mode: Arctic Shift primary, RSS failsafe."""
    subreddit = config["subreddit"]
    match_thread_patterns = config["match_thread_patterns"]
    media_config = config["media"]
    as_config = config.get("arctic_shift", {})

    logger.info("Running in dual-source mode (Arctic Shift + RSS)")

    # Determine time window: last polling interval
    interval_minutes = config.get("polling_interval_minutes", 5)
    now_ts = int(datetime.now(timezone.utc).timestamp())
    window_seconds = interval_minutes * 60 + 120  # add 2-min buffer for overlap
    after_ts = now_ts - window_seconds

    # Try Arctic Shift first
    as_posts = {}
    as_success = False
    try:
        rate_limit = as_config.get("rate_limit_seconds", 1.0)
        with ArcticShiftClient(rate_limit=rate_limit) as as_client:
            raw_posts = as_client.get_posts(subreddit, after_ts, now_ts)
            for p in raw_posts:
                pid = p.get("id")
                if pid:
                    as_posts[pid] = p
            as_success = True
            logger.info(f"Arctic Shift returned {len(as_posts)} posts")
    except Exception as e:
        logger.warning(f"Arctic Shift failed: {e}. Falling back to RSS only.")

    # Also fetch RSS as failsafe
    rss_posts = {}
    rss_success = False
    try:
        rss_client = create_client({**config, "mode": "rss"})
        raw_rss = rss_client.get_new_posts_rss(subreddit, limit=100)
        for p in raw_rss:
            pid = p.get("id")
            if pid:
                rss_posts[pid] = p
        rss_success = True
        logger.info(f"RSS returned {len(rss_posts)} posts")
    except Exception as e:
        logger.warning(f"RSS failed: {e}")

    if not as_success and not rss_success:
        logger.error("Both Arctic Shift and RSS failed. No posts collected.")
        return

    # Merge: union of all post IDs, AS data takes priority
    all_post_ids = set(as_posts.keys()) | set(rss_posts.keys())

    new_posts_count = 0
    media_downloaded = 0
    media_verified = 0

    for post_id in all_post_ids:
        if storage.is_seen(post_id):
            continue

        storage.mark_seen(post_id)

        if post_id in as_posts and post_id in rss_posts:
            post = extract_post_arctic_shift(as_posts[post_id], batch_id, match_thread_patterns)
            post = merge_arctic_shift_over_rss(post, rss_posts[post_id])
        elif post_id in as_posts:
            post = extract_post_arctic_shift(as_posts[post_id], batch_id, match_thread_patterns)
        else:
            post = extract_post_fields_rss(rss_posts[post_id], batch_id, match_thread_patterns)

        # Media processing — Arctic Shift posts have full media metadata
        if post_id in as_posts:
            urls, dl, vf = process_media(post, as_posts[post_id], storage, media_config, post_id)
        else:
            external_url = rss_posts[post_id].get("url", "")
            urls = []
            if external_url and external_url != rss_posts[post_id].get("permalink", ""):
                media_entry = {"url": external_url, "source": "external_link", "type": "link"}
                if media_config["verify_external_links"]:
                    result = verify_url(external_url)
                    media_entry.update(result)
                    vf = 1
                else:
                    vf = 0
                if is_reddit_hosted(external_url) and media_config["download_reddit_hosted"]:
                    save_dir = os.path.join(storage.media_dir, post_id)
                    result = download_media(
                        external_url, save_dir,
                        timeout=media_config["download_timeout_seconds"],
                        max_size_mb=media_config["max_file_size_mb"],
                    )
                    media_entry.update(result)
                    dl = 1 if result.get("downloaded") else 0
                else:
                    dl = 0
                urls.append(media_entry)
            else:
                dl, vf = 0, 0

        post["media_urls"] = urls
        media_downloaded += dl
        media_verified += vf

        storage.save_pending_post(batch_id, post_id, post)
        new_posts_count += 1

    storage.save_seen_posts()
    sources = []
    if as_success:
        sources.append("Arctic Shift")
    if rss_success:
        sources.append("RSS")
    logger.info(
        f"Scanner complete ({'+'.join(sources)}). {new_posts_count} new posts collected, "
        f"{media_downloaded} media downloaded, {media_verified} links verified."
    )


def run_rss(config, client, storage, logger, batch_id):
    subreddit = config["subreddit"]
    match_thread_patterns = config["match_thread_patterns"]
    media_config = config["media"]

    logger.info("Running in RSS mode (limited metadata)")
    posts = client.get_new_posts_rss(subreddit, limit=100)

    if not posts:
        logger.error("Failed to fetch posts or empty feed")
        return

    new_posts_count = 0
    media_downloaded = 0
    media_verified = 0

    for post_data in posts:
        post_id = post_data.get("id")
        if not post_id:
            continue

        if storage.is_seen(post_id):
            logger.info(f"Hit known post {post_id}. Stopping.")
            break

        storage.mark_seen(post_id)
        post = extract_post_fields_rss(post_data, batch_id, match_thread_patterns)

        external_url = post_data.get("url", "")
        if external_url and external_url != post_data.get("permalink", ""):
            media_entry = {"url": external_url, "source": "external_link", "type": "link"}
            if media_config["verify_external_links"]:
                result = verify_url(external_url)
                media_entry.update(result)
                media_verified += 1
            if is_reddit_hosted(external_url) and media_config["download_reddit_hosted"]:
                save_dir = os.path.join(storage.media_dir, post_id)
                result = download_media(
                    external_url, save_dir,
                    timeout=media_config["download_timeout_seconds"],
                    max_size_mb=media_config["max_file_size_mb"],
                )
                media_entry.update(result)
                if result.get("downloaded"):
                    media_downloaded += 1
            post["media_urls"].append(media_entry)

        storage.save_pending_post(batch_id, post_id, post)
        new_posts_count += 1

    storage.save_seen_posts()
    logger.info(
        f"Scanner complete (RSS). {new_posts_count} new posts collected, "
        f"{media_downloaded} media downloaded, {media_verified} links verified."
    )


def run_oauth(config, client, storage, logger, batch_id):
    subreddit = config["subreddit"]
    match_thread_patterns = config["match_thread_patterns"]
    media_config = config["media"]

    logger.info("Running in OAuth mode (full metadata)")

    new_posts_count = 0
    media_downloaded = 0
    media_verified = 0
    after = None
    stop = False

    while not stop:
        logger.info(f"Fetching posts (after={after})")
        response = client.get_new_posts_json(subreddit, limit=100, after=after)

        if not response or "data" not in response:
            logger.error("Failed to fetch posts or empty response")
            break

        children = response["data"].get("children", [])
        if not children:
            logger.info("No more posts to fetch")
            break

        for child in children:
            post_data = child.get("data", {})
            post_id = post_data.get("id")

            if not post_id:
                continue

            if storage.is_seen(post_id):
                logger.info(f"Hit known post {post_id}. Stopping.")
                stop = True
                break

            storage.mark_seen(post_id)
            post = extract_post_fields_oauth(post_data, batch_id, match_thread_patterns)

            urls, dl, vf = process_media(post, post_data, storage, media_config, post_id)
            post["media_urls"] = urls
            media_downloaded += dl
            media_verified += vf

            storage.save_pending_post(batch_id, post_id, post)
            new_posts_count += 1

        after = response["data"].get("after")
        if not after:
            logger.info("No more pages to fetch")
            break

    storage.save_seen_posts()
    logger.info(
        f"Scanner complete (OAuth). {new_posts_count} new posts collected, "
        f"{media_downloaded} media downloaded, {media_verified} links verified."
    )


def run():
    config = load_config()
    logger = setup_logging("scanner", config["storage"]["log_dir"])

    now = datetime.now(timezone.utc)
    end_date = datetime.fromisoformat(config["collection_end_date"])
    if now > end_date:
        logger.info("Collection end date has passed. Exiting.")
        return

    batch_id = now.strftime("%Y-%m-%dT%H:%M")
    logger.info(f"Scanner starting. Batch: {batch_id}")

    mode = config.get("mode", "rss")
    storage = Storage(config["storage"]["data_dir"])

    if mode == "arctic_shift+rss":
        run_arctic_shift_rss(config, storage, logger, batch_id)
    elif mode == "oauth":
        client = create_client(config)
        run_oauth(config, client, storage, logger, batch_id)
    else:
        client = create_client(config)
        run_rss(config, client, storage, logger, batch_id)


if __name__ == "__main__":
    run()
