"""
Sync backfill data from GitHub Release assets to a local directory.

Downloads compressed chunks from GitHub Releases, decompresses them,
and merges into final JSONL files in the specified output directory.

Usage:
  python sync_backfill.py                                    # default output
  python sync_backfill.py --output ~/OneDrive/historical-data
  python sync_backfill.py --repo alannemi/fifa-reddit-communities
"""

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile


DEFAULT_REPO = "alannemi/fifa-reddit-communities"
DEFAULT_OUTPUT = os.path.join(
    os.path.expanduser("~"),
    "Library", "CloudStorage", "OneDrive-McGillUniversity",
    "Research", "FIFA Chapter", "historical-data"
)


def get_releases(repo: str) -> list:
    """List all releases with backfill assets using gh CLI."""
    result = subprocess.run(
        ["gh", "release", "list", "--repo", repo, "--limit", "100", "--json",
         "tagName,name,assets"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error listing releases: {result.stderr}")
        sys.exit(1)
    return json.loads(result.stdout)


def download_asset(repo: str, tag: str, asset_name: str, dest_path: str):
    """Download a single release asset using gh CLI."""
    result = subprocess.run(
        ["gh", "release", "download", tag, "--repo", repo,
         "--pattern", asset_name, "--dir", os.path.dirname(dest_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  Error downloading {asset_name}: {result.stderr}")
        return False
    return True


def decompress_and_append(gz_path: str, output_path: str) -> int:
    """Decompress a .jsonl.gz file and append lines to the output file."""
    count = 0
    with gzip.open(gz_path, "rt", encoding="utf-8") as gz:
        with open(output_path, "a", encoding="utf-8") as out:
            for line in gz:
                out.write(line)
                count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Sync backfill data from GitHub Releases")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo (owner/name)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    progress_file = os.path.join(output_dir, "sync_progress.json")
    progress = {}
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            progress = json.load(f)

    synced_assets = set(progress.get("synced_assets", []))

    print(f"Fetching releases from {args.repo}...")
    releases = get_releases(args.repo)

    backfill_releases = [
        r for r in releases
        if r.get("tagName", "").startswith("backfill-")
    ]

    if not backfill_releases:
        print("No backfill releases found.")
        return

    print(f"Found {len(backfill_releases)} backfill releases")

    posts_output = os.path.join(output_dir, "soccer_posts.jsonl")
    comments_output = os.path.join(output_dir, "soccer_comments.jsonl")

    total_new_posts = 0
    total_new_comments = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for release in sorted(backfill_releases, key=lambda r: r["tagName"]):
            tag = release["tagName"]
            assets = release.get("assets", [])

            for asset in assets:
                asset_name = asset.get("name", "")
                if asset_name in synced_assets:
                    print(f"  Skipping {asset_name} (already synced)")
                    continue

                if not asset_name.endswith(".jsonl.gz"):
                    continue

                print(f"  Downloading {asset_name} from {tag}...")
                gz_path = os.path.join(tmpdir, asset_name)
                if not download_asset(args.repo, tag, asset_name, gz_path):
                    continue

                # Determine output target
                if "posts" in asset_name:
                    target = posts_output
                    label = "posts"
                elif "comments" in asset_name:
                    target = comments_output
                    label = "comments"
                else:
                    continue

                print(f"  Decompressing {asset_name}...")
                count = decompress_and_append(gz_path, target)
                print(f"  Appended {count:,} {label}")

                if label == "posts":
                    total_new_posts += count
                else:
                    total_new_comments += count

                synced_assets.add(asset_name)
                os.remove(gz_path)

    # Save progress
    progress["synced_assets"] = sorted(synced_assets)
    progress["total_posts"] = progress.get("total_posts", 0) + total_new_posts
    progress["total_comments"] = progress.get("total_comments", 0) + total_new_comments
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)

    print(f"\nSync complete!")
    print(f"  New posts synced: {total_new_posts:,}")
    print(f"  New comments synced: {total_new_comments:,}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
