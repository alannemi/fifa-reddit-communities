# r/soccer Data Collector

Academic research tool that collects public posts and comments from r/soccer using a two-agent architecture:

- **Scanner (Agent 1):** Runs hourly, collects new posts with full metadata and media
- **Harvester (Agent 2):** Runs hourly with a 48-hour delay, collects comment trees and updated post metadata

## Setup

1. Create a private GitHub repository
2. Push this code to the repository
3. GitHub Actions will automatically start running on the cron schedule

## Local Export

To convert collected JSON data into CSV files for R analysis:

```bash
pip install -r requirements.txt
python -m src.export
```

Output files in `data/export/`:
- `posts.csv` — one row per post with t0 and t48 snapshots
- `comments.csv` — one row per comment, linked by post_id
- `collection_summary.csv` — batch-level statistics

## Configuration

All parameters are in `config.yaml`. Key settings:
- `subreddit`: target subreddit (default: soccer)
- `collection_end_date`: when to stop collecting
- `comment_limits.default`: max comments per regular post (10,000)
- `comment_limits.match_thread`: max comments for match threads (2,500)

## Monitoring

- Check the **Actions** tab in GitHub to see run status
- Browse `logs/` for detailed per-run logs
- Check `data/index/truncated.json` for posts that hit the comment ceiling
