##############################################################################
# FIFA COMMUNITIES CHAPTER
## 4. RENDER THE CODEBOOK  (Markdown -> HTML -> PDF)
##############################################################################
#
# The codebook's source of truth is data/codebook_soccer_posts.md (plain
# Markdown so it can be diffed and edited by anyone).  This helper renders it
# to data/codebook_soccer_posts.pdf for sharing with reviewers.
#
# Toolchain (chosen because pandoc/quarto are not installed on this machine):
#   commonmark (R package, GitHub-flavoured tables enabled)  -> HTML
#   Google Chrome in headless mode                           -> PDF
# Styling comes from docs/codebook.css.
#
# HOW TO RUN:  Rscript code/04-render-codebook.R   (from the repository root)
##############################################################################

find_root <- function() {
  if (file.exists("data/codebook_soccer_posts.md")) return(normalizePath("."))
  cand <- path.expand("~/OneDrive - McGill University/SHARED FOLDER - FIFA Chapter/fifa-reddit-communities")
  if (file.exists(file.path(cand, "data/codebook_soccer_posts.md"))) return(cand)
  stop("repository root not found")
}
setwd(find_root())

md   <- readLines("data/codebook_soccer_posts.md", encoding = "UTF-8")
html <- commonmark::markdown_html(paste(md, collapse = "\n"), extensions = TRUE, smart = FALSE)
css  <- readLines("docs/codebook.css", encoding = "UTF-8")
page <- c("<!doctype html><html><head><meta charset=\"utf-8\">",
          "<title>Codebook: r/soccer post corpus</title>",
          "<style>", css, "</style></head><body>", html, "</body></html>")
html_path <- file.path(tempdir(), "codebook_soccer_posts.html")
writeLines(page, html_path, useBytes = TRUE)

chrome <- "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if (!file.exists(chrome)) stop("Google Chrome not found at ", chrome, " — adjust the path or use another HTML-to-PDF tool.")
pdf_path <- normalizePath("data/codebook_soccer_posts.pdf", mustWork = FALSE)
status <- system2(chrome, c("--headless", "--disable-gpu", "--no-pdf-header-footer",
                            shQuote(paste0("--print-to-pdf=", pdf_path)), shQuote(html_path)),
                  stdout = FALSE, stderr = FALSE)
if (status != 0 || !file.exists(pdf_path)) stop("Chrome PDF export failed (status ", status, ")")
cat("Rendered", pdf_path, "(", round(file.size(pdf_path) / 1024), "KB )\n")
