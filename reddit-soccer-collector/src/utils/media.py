import os
import requests
from urllib.parse import urlparse


REDDIT_MEDIA_DOMAINS = {"i.redd.it", "v.redd.it", "preview.redd.it", "i.imgur.com"}


def extract_media_urls(post_data: dict) -> list:
    """Extract all media URLs from a Reddit post's JSON data."""
    urls = []

    post_url = post_data.get("url", "")
    domain = post_data.get("domain", "")

    if post_data.get("is_video") and "media" in post_data and post_data["media"]:
        reddit_video = post_data["media"].get("reddit_video", {})
        if reddit_video.get("fallback_url"):
            urls.append({
                "url": reddit_video["fallback_url"],
                "source": "reddit_video",
                "type": "video",
            })

    if "preview" in post_data and post_data["preview"]:
        images = post_data["preview"].get("images", [])
        for img in images:
            source = img.get("source", {})
            if source.get("url"):
                urls.append({
                    "url": source["url"],
                    "source": "preview",
                    "type": "image",
                })

    if "gallery_data" in post_data and post_data["gallery_data"]:
        media_metadata = post_data.get("media_metadata", {})
        for item in post_data["gallery_data"].get("items", []):
            media_id = item.get("media_id", "")
            if media_id in media_metadata:
                meta = media_metadata[media_id]
                if meta.get("s", {}).get("u"):
                    urls.append({
                        "url": meta["s"]["u"],
                        "source": "gallery",
                        "type": meta.get("e", "image").lower(),
                    })

    parsed = urlparse(post_url)
    if parsed.hostname in REDDIT_MEDIA_DOMAINS:
        if not any(u["url"] == post_url for u in urls):
            urls.append({
                "url": post_url,
                "source": "post_url",
                "type": "image" if "i." in parsed.hostname else "video",
            })

    if post_url and not post_data.get("is_self", True) and not any(u["url"] == post_url for u in urls):
        urls.append({
            "url": post_url,
            "source": "external_link",
            "type": "link",
        })

    return urls


def is_reddit_hosted(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in REDDIT_MEDIA_DOMAINS


def download_media(url: str, save_dir: str, timeout: int = 30, max_size_mb: int = 100) -> dict:
    """Download a media file from a URL. Returns metadata about the download."""
    os.makedirs(save_dir, exist_ok=True)

    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            return {
                "downloaded": False,
                "reason": f"file exceeds {max_size_mb}MB limit",
                "content_length": int(content_length),
            }

        content_type = response.headers.get("Content-Type", "")
        ext = _guess_extension(url, content_type)
        filename = f"media_{hash(url) % 10**8:08d}{ext}"
        filepath = os.path.join(save_dir, filename)

        total_size = 0
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                total_size += len(chunk)
                if total_size > max_size_mb * 1024 * 1024:
                    f.close()
                    os.remove(filepath)
                    return {
                        "downloaded": False,
                        "reason": f"file exceeds {max_size_mb}MB during download",
                    }
                f.write(chunk)

        return {
            "downloaded": True,
            "local_path": filepath,
            "file_size_bytes": total_size,
            "content_type": content_type,
        }

    except requests.exceptions.RequestException as e:
        return {"downloaded": False, "reason": str(e)}


def verify_url(url: str, timeout: int = 5) -> dict:
    """Send a HEAD request to verify a URL is accessible."""
    try:
        response = requests.head(url, timeout=timeout, allow_redirects=True)
        return {
            "verified": response.status_code == 200,
            "status_code": response.status_code,
            "content_type": response.headers.get("Content-Type", ""),
            "content_length": response.headers.get("Content-Length", ""),
            "final_url": response.url,
        }
    except requests.exceptions.RequestException:
        return {"verified": False, "status_code": None, "error": "unreachable"}


def _guess_extension(url: str, content_type: str) -> str:
    ext_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "video/mp4": ".mp4",
    }
    for ct, ext in ext_map.items():
        if ct in content_type:
            return ext

    parsed = urlparse(url)
    path = parsed.path.lower()
    for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov"]:
        if path.endswith(ext):
            return ext

    return ".bin"
