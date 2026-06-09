import json
import os


class Storage:
    """Handles file I/O, indexing, and deduplication."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.pending_dir = os.path.join(data_dir, "pending")
        self.collected_dir = os.path.join(data_dir, "collected")
        self.index_dir = os.path.join(data_dir, "index")
        self.media_dir = os.path.join(data_dir, "media")
        self.export_dir = os.path.join(data_dir, "export")

        for d in [self.pending_dir, self.collected_dir, self.index_dir, self.media_dir, self.export_dir]:
            os.makedirs(d, exist_ok=True)

        self.seen_posts_file = os.path.join(self.index_dir, "seen_posts.json")
        self.truncated_file = os.path.join(self.index_dir, "truncated.json")
        self.seen_posts = self._load_seen_posts()

    def _load_seen_posts(self) -> set:
        if os.path.exists(self.seen_posts_file):
            with open(self.seen_posts_file, "r") as f:
                return set(json.load(f))
        return set()

    def save_seen_posts(self):
        with open(self.seen_posts_file, "w") as f:
            json.dump(sorted(self.seen_posts), f)

    def is_seen(self, post_id: str) -> bool:
        return post_id in self.seen_posts

    def mark_seen(self, post_id: str):
        self.seen_posts.add(post_id)

    def save_pending_post(self, batch_id: str, post_id: str, data: dict):
        batch_dir = os.path.join(self.pending_dir, batch_id)
        os.makedirs(batch_dir, exist_ok=True)
        filepath = os.path.join(batch_dir, f"post_{post_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_collected_post(self, batch_id: str, post_id: str, data: dict):
        # Group by date (YYYY-MM-DD) instead of hourly batch
        date_folder = batch_id[:10]  # "2026-06-07T20" -> "2026-06-07"
        date_dir = os.path.join(self.collected_dir, date_folder)
        os.makedirs(date_dir, exist_ok=True)
        filepath = os.path.join(date_dir, f"post_{post_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def get_pending_batch(self, batch_id: str) -> list:
        batch_dir = os.path.join(self.pending_dir, batch_id)
        if not os.path.exists(batch_dir):
            return []
        posts = []
        for filename in os.listdir(batch_dir):
            if filename.startswith("post_") and filename.endswith(".json"):
                filepath = os.path.join(batch_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    posts.append(json.load(f))
        return posts

    def remove_pending_post(self, batch_id: str, post_id: str):
        batch_dir = os.path.join(self.pending_dir, batch_id)
        filepath = os.path.join(batch_dir, f"post_{post_id}.json")
        if os.path.exists(filepath):
            os.remove(filepath)
        if os.path.exists(batch_dir) and not os.listdir(batch_dir):
            os.rmdir(batch_dir)

    def log_truncated(self, post_id: str, limit_applied: int, comments_collected: int):
        truncated = []
        if os.path.exists(self.truncated_file):
            with open(self.truncated_file, "r") as f:
                truncated = json.load(f)
        truncated.append({
            "post_id": post_id,
            "limit_applied": limit_applied,
            "comments_collected": comments_collected,
        })
        with open(self.truncated_file, "w") as f:
            json.dump(truncated, f, indent=2)

    def get_all_collected(self) -> list:
        posts = []
        if not os.path.exists(self.collected_dir):
            return posts
        for batch_name in sorted(os.listdir(self.collected_dir)):
            batch_dir = os.path.join(self.collected_dir, batch_name)
            if not os.path.isdir(batch_dir):
                continue
            for filename in sorted(os.listdir(batch_dir)):
                if filename.startswith("post_") and filename.endswith(".json"):
                    filepath = os.path.join(batch_dir, filename)
                    with open(filepath, "r", encoding="utf-8") as f:
                        posts.append(json.load(f))
        return posts
