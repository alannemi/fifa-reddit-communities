import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html import unescape

import requests


class RedditClient:
    """HTTP client for Reddit. Supports RSS (no auth) and OAuth modes."""

    def __init__(self, user_agent: str, requests_per_minute: int = 8, mode: str = "rss",
                 client_id: str = None, client_secret: str = None,
                 username: str = None, password: str = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0.0
        self.mode = mode

        if mode == "oauth" and all([client_id, client_secret, username, password]):
            self._authenticate(client_id, client_secret, username, password)

    def _authenticate(self, client_id, client_secret, username, password):
        auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
        data = {
            "grant_type": "password",
            "username": username,
            "password": password,
        }
        response = requests.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth, data=data,
            headers={"User-Agent": self.session.headers["User-Agent"]},
        )
        response.raise_for_status()
        token = response.json()["access_token"]
        self.session.headers.update({"Authorization": f"Bearer {token}"})
        self.base_url = "https://oauth.reddit.com"

    def _throttle(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()

    def _get(self, url: str, params: dict = None, retries: int = 3, backoff: int = 30):
        for attempt in range(retries):
            self._throttle()
            try:
                response = self.session.get(url, params=params, timeout=30)
                if response.status_code == 200:
                    return response
                if response.status_code == 429:
                    wait = backoff * (attempt + 1)
                    time.sleep(wait)
                    continue
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff * (attempt + 1))
        return None

    def get_json(self, url: str, params: dict = None) -> dict:
        response = self._get(url, params=params)
        if response:
            return response.json()
        return None

    # ── RSS methods (no auth required) ──

    def get_new_posts_rss(self, subreddit: str, limit: int = 100) -> list:
        url = f"https://www.reddit.com/r/{subreddit}/new/.rss"
        params = {"limit": limit}
        response = self._get(url, params=params)
        if not response:
            return []
        return self._parse_rss_feed(response.text)

    def _parse_rss_feed(self, xml_text: str) -> list:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        posts = []

        for entry in root.findall("atom:entry", ns):
            post_id_raw = entry.find("atom:id", ns)
            if post_id_raw is None:
                continue
            post_id = post_id_raw.text.replace("t3_", "")

            title_el = entry.find("atom:title", ns)
            author_el = entry.find("atom:author/atom:name", ns)
            link_el = entry.find("atom:link", ns)
            published_el = entry.find("atom:published", ns)
            updated_el = entry.find("atom:updated", ns)
            content_el = entry.find("atom:content", ns)

            content_html = ""
            if content_el is not None and content_el.text:
                content_html = unescape(content_el.text)

            link_url = ""
            permalink = ""
            if link_el is not None:
                permalink = link_el.get("href", "")

            external_url = self._extract_link_from_content(content_html)

            published_str = published_el.text if published_el is not None else ""
            created_utc = None
            created_datetime = None
            if published_str:
                try:
                    dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    created_utc = dt.timestamp()
                    created_datetime = dt.isoformat()
                except ValueError:
                    pass

            selftext_html = self._extract_selftext_from_content(content_html)

            posts.append({
                "id": post_id,
                "name": f"t3_{post_id}",
                "title": title_el.text if title_el is not None else "",
                "author": (author_el.text or "").replace("/u/", "") if author_el is not None else "[unknown]",
                "permalink": permalink,
                "url": external_url or permalink,
                "created_utc": created_utc,
                "created_datetime": created_datetime,
                "selftext_html": selftext_html,
                "selftext": "",
                "content_html_raw": content_html,
                "source": "rss",
            })

        return posts

    def _extract_link_from_content(self, html: str) -> str:
        import re
        match = re.search(r'\[link\]\s*</a>\s*</span>', html)
        if match:
            link_match = re.search(r'<a href="([^"]+)">\s*\[link\]', html)
            if link_match:
                return link_match.group(1)
        return ""

    def _extract_selftext_from_content(self, html: str) -> str:
        import re
        match = re.search(r'<!-- SC_OFF -->(.*?)<!-- SC_ON -->', html, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    # ── OAuth/JSON methods (requires auth) ──

    def get_new_posts_json(self, subreddit: str, limit: int = 100, after: str = None) -> dict:
        url = f"{self.base_url}/r/{subreddit}/new.json"
        params = {"limit": limit, "raw_json": 1}
        if after:
            params["after"] = after
        return self.get_json(url, params=params)

    def get_post_with_comments(self, subreddit: str, post_id: str, limit: int = 500, sort: str = "confidence") -> list:
        if self.mode == "oauth":
            url = f"{self.base_url}/r/{subreddit}/comments/{post_id}.json"
        else:
            url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
        params = {"limit": limit, "sort": sort, "raw_json": 1}
        return self.get_json(url, params=params)

    def get_more_comments(self, link_id: str, children: list, sort: str = "confidence") -> dict:
        if self.mode == "oauth":
            url = f"{self.base_url}/api/morechildren.json"
        else:
            url = "https://www.reddit.com/api/morechildren.json"
        params = {
            "api_type": "json",
            "link_id": link_id,
            "children": ",".join(children),
            "sort": sort,
            "raw_json": 1,
        }
        return self.get_json(url, params=params)

    def get_post_rss(self, subreddit: str, post_id: str) -> dict:
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/.rss"
        params = {"limit": 500}
        response = self._get(url, params=params)
        if not response:
            return None
        return self._parse_comments_rss(response.text, post_id)

    def _parse_comments_rss(self, xml_text: str, post_id: str) -> dict:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        comments = []

        for entry in root.findall("atom:entry", ns):
            entry_id = entry.find("atom:id", ns)
            if entry_id is None:
                continue
            raw_id = entry_id.text
            if raw_id.startswith("t3_"):
                continue

            comment_id = raw_id.replace("t1_", "")
            author_el = entry.find("atom:author/atom:name", ns)
            content_el = entry.find("atom:content", ns)
            published_el = entry.find("atom:published", ns)
            updated_el = entry.find("atom:updated", ns)

            published_str = published_el.text if published_el is not None else ""
            created_utc = None
            created_datetime = None
            if published_str:
                try:
                    dt = datetime.fromisoformat(published_str.replace("Z", "+00:00"))
                    created_utc = dt.timestamp()
                    created_datetime = dt.isoformat()
                except ValueError:
                    pass

            body_html = ""
            if content_el is not None and content_el.text:
                body_html = unescape(content_el.text)

            comments.append({
                "id": comment_id,
                "name": f"t1_{comment_id}",
                "author": (author_el.text or "").replace("/u/", "") if author_el is not None else "[unknown]",
                "body": "",
                "body_html": body_html,
                "created_utc": created_utc,
                "created_datetime": created_datetime,
                "score": 0,
                "ups": 0,
                "downs": 0,
                "parent_id": f"t3_{post_id}",
                "depth": 0,
                "author_flair_text": "",
                "author_flair_css_class": "",
                "author_fullname": "",
                "edited": False,
                "is_submitter": False,
                "distinguished": None,
                "stickied": False,
                "controversiality": 0,
                "total_awards_received": 0,
                "all_awardings": [],
                "source": "rss",
            })

        return {"post_id": post_id, "comments": comments}
