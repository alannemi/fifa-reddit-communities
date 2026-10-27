# Sentiment coding guide: validating VADER on 337 r/soccer posts

*For the two co-authors who code the sentiment validation sample. Session 09, 2026-10-27; decision E1 (Alan Nemirovski).*

## Why this coding matters

Every post in the corpus has a VADER sentiment score (`post_sentiment`). VADER is a general-purpose word list with rules, and it is known to misread sarcasm and football vocabulary ("brutal tackle", "killer pass", "sick goal", "crash out"). Your codes are the human reference. We measure two things from them:

1. how often the two of you agree;
2. how often VADER agrees with you.

Both numbers go into the codebook (§8.2) and the chapter's methods section.

## Files

- **Your sheet.** Coder A: `results/analysis/manual review/sentiment_coding/coder_A.csv`; coder B: `coder_B.csv`. Alan assigns who is A and who is B.
- **What the two sheets hold.** The same 337 posts in the same random order. The posts come from all nine period bins and include both link posts and self posts.
- **Columns:**
  - `id`: the post id;
  - `title`: the post title, the text you code;
  - `body_excerpt`: for self posts, the first 1,000 characters of the body; empty otherwise;
  - `is_self`;
  - `code`: you fill this in;
  - `notes`: optional.

**Code independently.** Do not look at the other coder's sheet, and do not discuss individual posts until both sheets are returned.

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

## Returning the sheet

Upload your filled sheet to the GitHub repository under the **same file name**, in `results/analysis/manual review/sentiment_coding/`. Upload it through GitHub's web interface, as with the review files. Keep the columns and the row order as they are, and save as CSV (UTF-8).

## After both sheets are back

1. **Run the comparison.** Alan or Claude runs `Rscript code/10-sentiment-validation.R`. It reports coder-to-coder agreement and Cohen's κ, then each coder against VADER.
2. **Adjudicate the disagreements.** The two coders, or Alan, settle the posts where the codes differ. The result goes in an `adjudicated.csv` (columns `id`, `code`) in the same folder.
3. **Rerun the comparison.** The script then compares the agreed reference with VADER: a confusion matrix, precision and recall per class, and agreement by period bin and by link/self post.
