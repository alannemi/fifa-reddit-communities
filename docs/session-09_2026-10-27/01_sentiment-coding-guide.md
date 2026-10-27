# Sentiment coding guide: validating VADER on 337 r/soccer posts

*For the coder of the sentiment validation sample. Session 09, 2026-10-27; decision E1 (Alan Nemirovski). Revised the same day: **one coder, Alan Nemirovski**, codes the sample, so there is no second sheet and no adjudication.*

## Why this coding matters

Every post in the corpus has a VADER sentiment score (`post_sentiment`). VADER is a general-purpose word list with rules, and it is known to misread sarcasm and football vocabulary ("brutal tackle", "killer pass", "sick goal", "crash out"). Your codes are the human reference. From them we measure how often VADER agrees with a human reader: percent agreement, Cohen's κ, which kinds of posts it gets wrong, and whether that differs between periods or between link and self posts. These numbers go into the codebook (§8.2) and the chapter's methods section. With one coder, the reliability of the human reference itself is not measured; the codebook states this.

## Files

- **Your sheet:** `results/analysis/manual review/sentiment_coding/coder_A.csv`, the only sheet.
- **What the two sheets hold.** The same 337 posts in the same random order. The posts come from all nine period bins and include both link posts and self posts.
- **Columns:**
  - `id`: the post id;
  - `title`: the post title, the text you code;
  - `body_excerpt`: for self posts, the first 1,000 characters of the body; empty otherwise;
  - `is_self`;
  - `code`: you fill this in;
  - `notes`: optional.

**Code blind.** Do not look at the posts' VADER scores (`data/analysis/sentiment_validation_sample.csv`, `post_variables.csv` or the figures' tables) while coding, and do not run the validation script until the sheet is finished.

## What to code

Put one of three values in `code`, describing the **sentiment expressed in the title**:

| code | Use when the title… | Examples |
|---|---|---|
| `positive` | expresses or clearly conveys approval, joy, admiration, praise, celebration, hope or excitement | "What a goal by Messi", "Congratulations to Spain, deserved champions", "Can't wait for the final!" |
| `negative` | expresses or clearly conveys disapproval, anger, sadness, criticism, mockery, fear or complaint | "Worst refereeing I've ever seen", "RIP to a legend", "This VAR decision is a joke" |
| `neutral` | reports information or asks a question without an evaluative stance, or the stance is too mixed or unclear to call | "Official: Ronaldo joins Juventus", "Line-ups for Brazil vs Germany", "Who was your player of the tournament?" |

### Rules
1. **Code the title.** The body excerpt is shown only as context, for example to recognise sarcasm or a reference you do not know. Do not code the sentiment of the body itself.
2. **Code what the author conveys, not the event.** A neutral report of a sad event ("Player X has died aged 34") is `neutral` unless the title adds a stance. "RIP Player X, a legend" is `negative` (grief). "Germany 1–7 Brazil" as a bare score is `neutral`.
3. **Sarcasm and irony.** Code the intended meaning. "Great defending there, lads" under a defensive error is `negative`.
4. **Football vocabulary.** Words that sound violent or negative are often praise in football: "killer pass", "sick goal", "absolutely destroyed them", "brutal free kick". Code them as the fan means them. "Crash out", "bottled it" and "humiliated" are negative for the team concerned.
5. **Mixed or unclear stance.** If the title is genuinely balanced, or you cannot tell, use `neutral` and write a word in `notes` (for example "mixed" or "unclear").
6. **Non-English titles.** Code them if you can read them. Otherwise leave `code` blank and write "language" in `notes`.
7. **Do not research.** Do not look posts up on Reddit. Code from what the sheet shows. A first reading should take 10–20 seconds per post, about 1.5 hours for the sheet.

Spelling: write exactly `positive`, `neutral` or `negative`. The script also accepts `pos`, `neu`, `neg` and `1`, `0`, `-1`.

## Opening and saving the sheet

The titles contain accented and non-Latin characters (45 posts), and some bodies run over several lines (43 posts). Open and save the file so that these survive:

1. **Excel (Mac or Windows).** Do not double-click the file. In a blank workbook, choose **Data → From Text/CSV** (Mac: **File → Import → CSV file**). Set the file origin to **Unicode (UTF-8)** and the delimiter to **comma**, and import every column as **Text**.
2. **Make it readable.** Turn on **Wrap Text** for the `title` and `body_excerpt` columns, and widen them.
3. **Optional: a drop-down for `code`.** Select the `code` column, then **Data → Data Validation → List**: `positive,neutral,negative`.
4. **Sorting is allowed.** The script matches rows by `id`, so the order does not matter. Do not delete or edit rows, `id` values or column names.
5. **Save as `CSV UTF-8 (Comma delimited) (.csv)`** under the same name. Do not use plain "CSV", which loses the accents, and do not save as `.xlsx`.

Google Sheets also works: File → Import → Upload, then File → Download → Comma-separated values.

## Checking your sheet

After saving, `Rscript code/10-sentiment-validation.R` (run by Alan or Claude) reads whatever has been coded. It stops with a message if a code is misspelled, and it reports how many rows are coded. **Run it only after you have finished coding**, because its output includes agreement with VADER.

## Returning the sheet

Save the filled sheet in place, under the same name (`coder_A.csv`), as CSV UTF-8. Then say it is finished: Claude runs the check and the comparison and commits the sheet with the results. Uploading through GitHub's web interface also works, under the same name and in the same folder.

## After the sheet is finished

Run `Rscript code/10-sentiment-validation.R` (Alan or Claude). With only `coder_A.csv` in the folder, the script treats your codes as the reference and reports, against VADER:
- percent agreement and Cohen's κ, with a bootstrap 95 % interval;
- the confusion matrix;
- precision, recall and F1 per class;
- agreement by period bin and by link/self post.

The results go to `results/descriptives/sentiment_validation_*.csv`, and from there into codebook §8.2.
