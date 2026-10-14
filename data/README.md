# Why the data files are not on GitHub

The corpus in this folder is about 14 GB (raw export 3.1 GB, clean master 1.9 GB, yearly files 2.4 GB,
side files 0.7 GB, raw JSONL batches 7 GB). GitHub limits individual files to 100 MB and repositories to a
few GB, so **the data is shared through the project's OneDrive folder**
(`SHARED FOLDER - FIFA Chapter/fifa-reddit-communities/data/`), not through git.

If you cloned this repository from GitHub:

1. Copy the whole `data/` folder from OneDrive into the repository root (next to `code/`).
2. Open `data/codebook_soccer_posts.md` (the only data-folder files kept in git are this note and the
   codebook, `.md` and `.pdf`).
3. Load the CSVs with every column as text — see the codebook §5 or the root README §2.

The data can also be rebuilt from the raw export with `code/02-data-cleaning.R`; the raw export itself can
be re-collected from the Arctic Shift archive with `reddit-soccer-collector/src/fetch_posts_range.py`.
