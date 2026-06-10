"""
Arctic Shift API client for retrieving Reddit data from the community-run archive.

Arctic Shift (arctic-shift.photon-reddit.com) provides full Reddit post and comment
data without authentication. Pagination uses a time-cursor approach: walk backwards
through created_utc timestamps, fetching batches of up to 100 items.
"""

import time

import httpx


BASE_URL = "https://arctic-shift.photon-reddit.com/api"
PAGE_SIZE = 100
MAX_RETRIES = 5
RATE_LIMIT = 1.0

HEADERS = {
    "User-Agent": "MEO-Research/1.0 (McGill University; academic research)"
}


class ArcticShiftClient:

    def __init__(self, rate_limit: float = RATE_LIMIT):
        self.rate_limit = rate_limit
        self.last_request_time = 0.0
        self.client = httpx.Client(headers=HEADERS, timeout=60)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _throttle(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _fetch_page(self, endpoint: str, subreddit: str,
                    after_ts: int, before_ts: int) -> list:
        attempt = 0
        while True:
            self._throttle()
            try:
                resp = self.client.get(f"{BASE_URL}/{endpoint}", params={
                    "subreddit": subreddit,
                    "after": after_ts,
                    "before": before_ts,
                    "limit": PAGE_SIZE,
                })
                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 30))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()["data"]
            except (httpx.RequestError, httpx.HTTPStatusError):
                attempt += 1
                if attempt > MAX_RETRIES:
                    raise
                backoff = min(2 ** attempt, 60)
                time.sleep(backoff)

    def get_posts(self, subreddit: str, after_ts: int, before_ts: int) -> list:
        """Fetch all posts in a time range, paginating backwards."""
        all_items = []
        cursor = before_ts

        while cursor > after_ts:
            batch = self._fetch_page("posts/search", subreddit, after_ts, cursor)
            if not batch:
                break
            all_items.extend(batch)
            oldest = min(item["created_utc"] for item in batch)
            if oldest >= cursor:
                break
            cursor = oldest

        return all_items

    def get_comments(self, subreddit: str, after_ts: int, before_ts: int) -> list:
        """Fetch all comments in a time range, paginating backwards."""
        all_items = []
        cursor = before_ts

        while cursor > after_ts:
            batch = self._fetch_page("comments/search", subreddit, after_ts, cursor)
            if not batch:
                break
            all_items.extend(batch)
            oldest = min(item["created_utc"] for item in batch)
            if oldest >= cursor:
                break
            cursor = oldest

        return all_items

    def get_comments_for_post(self, subreddit: str, post_id: str) -> list:
        """Fetch all comments for a specific post by searching with link_id."""
        all_items = []
        cursor = int(time.time()) + 86400

        while True:
            self._throttle()
            attempt = 0
            while True:
                try:
                    resp = self.client.get(f"{BASE_URL}/comments/search", params={
                        "subreddit": subreddit,
                        "link_id": f"t3_{post_id}",
                        "before": cursor,
                        "limit": PAGE_SIZE,
                    })
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 30))
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    batch = resp.json()["data"]
                    break
                except (httpx.RequestError, httpx.HTTPStatusError):
                    attempt += 1
                    if attempt > MAX_RETRIES:
                        raise
                    backoff = min(2 ** attempt, 60)
                    time.sleep(backoff)

            if not batch:
                break
            all_items.extend(batch)
            oldest = min(item["created_utc"] for item in batch)
            if oldest >= cursor:
                break
            cursor = oldest

        return all_items

    def get_post_by_id(self, subreddit: str, post_id: str) -> dict | None:
        """Fetch a single post by searching a narrow time window around it."""
        self._throttle()
        try:
            resp = self.client.get(f"{BASE_URL}/posts/search", params={
                "subreddit": subreddit,
                "ids": post_id,
                "limit": 1,
            })
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                time.sleep(wait)
                return self.get_post_by_id(subreddit, post_id)
            resp.raise_for_status()
            data = resp.json()["data"]
            return data[0] if data else None
        except (httpx.RequestError, httpx.HTTPStatusError):
            return None
