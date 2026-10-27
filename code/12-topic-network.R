##############################################################################
# FIFA COMMUNITIES CHAPTER
## 12. TOPIC NETWORK  (how the 6,245 topics of the nine period bins relate to one another)
##############################################################################
#
# PURPOSE
# -------
# A map of the whole corpus at the level of its unit of analysis, the topic
# (decision D7): which topics are about the same things, how they group into
# larger families, and whether those families recur across the period bins.
# Author's request, 2026-10-27: edges = semantic similarity, static figure.
#
# METHOD
# ------
# * Nodes: every labelled topic (results/analysis/topic_labels_all_bins.csv),
#   sized by its HDBSCAN cluster (`n_cluster`, independent of the outlier tiers).
# * Similarity: cosine of the topic embeddings (bin-N/topic_embeddings.npy, the
#   mean all-mpnet-base-v2 embedding of the topic's posts).  All nine models
#   embed into the same space, so topics of different bins can be compared.
# * Edges: each topic is joined to its K = 5 most similar topics in the whole
#   corpus, kept only if the similarity is at least 0.65 (about the 10th
#   percentile of the 5th neighbour's similarity; the top 1 % of all pairs).
#   Undirected; weight = similarity.
# * Families: Louvain community detection on the weighted graph (seed 42).
#   A family is labelled by its two largest topics, which name it only roughly.
# * Layout: Fruchterman-Reingold (seed 42) on the main connected network; topics
#   outside it (isolated, or in small groups linked only to each other) stretch
#   the layout, so they are counted in the caption and listed in the tables but
#   not drawn.  The view is framed on the central 98 % of positions (1st-99th
#   percentile of x and y): a few chains of weakly linked topics reach far out; the
#   caption counts the topics beyond the frame.
#
# HOW TO RUN (repository root; needs the .venv for reading the .npy arrays through reticulate)
# ----------------------------
#   Rscript code/12-topic-network.R          # ~1-2 min
#
# OUTPUTS  (results/descriptives/)
# -------
#   fig_D1_topic_network_families.png|pdf   the network coloured by the eight largest families, labelled F1-F8
#   fig_D1b_topic_network_families_unlabelled.png|pdf   the same without the labels on the map (legend only)
#   fig_D2_topic_network_periods.png|pdf    the same layout coloured by period bin
#   tab_D1_topic_network_nodes.csv          one row per topic: family, degree, layout position
#   tab_D1_topic_network_edges.csv          one row per edge: the two topics, similarity, same bin or not
#   tab_D1_topic_families.csv               one row per family: size, bins, largest topics, tones
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(igraph); library(ggrepel) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                                     # shared helpers and the bin table (code/R/pipeline-utils.R)
source(file.path(dirname(.utils), "plot-utils.R")) # palette, theme, save_fig / save_tab
ROOT <- find_root()
setwd(ROOT)

LABELS   <- "results/analysis/topic_labels_all_bins.csv"
OUT_DIR  <- "results/descriptives"
K        <- 5L                     # nearest neighbours per topic
MIN_SIM  <- 0.65                   # cosine cut-off for an edge
N_COLOUR <- 8L                     # families coloured (the categorical palette has eight slots)
SEED     <- 42L
reticulate::use_python(".venv/bin/python", required = TRUE)
np <- reticulate::import("numpy")

# ---------------------------------------------------------------------------
# 1. Nodes and their embeddings (row i of bin-N/topic_embeddings.npy = topic i of bin N)
# ---------------------------------------------------------------------------
lab <- read_chr(LABELS, col_select = c("period_bin", "topic_id", "topic_uid", "name", "tone_category", "n_cluster")) |>
  mutate(period_bin = as.integer(period_bin), topic_id = as.integer(topic_id), n_cluster = as.integer(n_cluster)) |>
  arrange(period_bin, topic_id)
emb <- do.call(rbind, lapply(1:9, function(b) {
  e <- np$load(sprintf("results/analysis/bin-%d/topic_embeddings.npy", b))
  ids <- lab$topic_id[lab$period_bin == b]
  check(sprintf("bin %d: one embedding per labelled topic", b), nrow(e) == length(ids) && identical(ids, seq_len(nrow(e)) - 1L),
        sprintf("%d", nrow(e)))
  e
}))
emb <- emb / sqrt(rowSums(emb^2))                  # unit length: the dot product is the cosine
log_msg("topics: %d, embedding dimension %d", nrow(emb), ncol(emb))

# ---------------------------------------------------------------------------
# 2. Edges: K nearest neighbours above the cut-off, undirected
# ---------------------------------------------------------------------------
S <- tcrossprod(emb); diag(S) <- -Inf
nn <- t(apply(S, 1, function(r) order(r, decreasing = TRUE)[seq_len(K)]))
edges <- tibble(i = rep(seq_len(nrow(S)), each = K), j = as.vector(t(nn))) |>
  mutate(similarity = S[cbind(i, j)]) |>
  filter(similarity >= MIN_SIM) |>
  transmute(a = pmin(i, j), b = pmax(i, j), similarity) |>
  distinct(a, b, .keep_all = TRUE)
rm(S); invisible(gc())
edges <- edges |> mutate(from = lab$topic_uid[a], to = lab$topic_uid[b],
                         same_bin = lab$period_bin[a] == lab$period_bin[b])
check("no self-loops, similarities within [cut-off, 1]", all(edges$a != edges$b) && all(edges$similarity >= MIN_SIM & edges$similarity <= 1 + 1e-9))
log_msg("edges: %d (%.1f %% between topics of different bins)", nrow(edges), 100 * mean(!edges$same_bin))

# ---------------------------------------------------------------------------
# 3. Graph, families (Louvain) and layout
# ---------------------------------------------------------------------------
g <- graph_from_data_frame(edges |> select(from, to, weight = similarity), directed = FALSE,
                           vertices = lab |> select(name = topic_uid))
set.seed(SEED)
fam <- cluster_louvain(g, weights = E(g)$weight)
lab$family_raw <- as.integer(membership(fam))
lab$degree <- as.integer(degree(g))
check("every topic in exactly one family", !anyNA(lab$family_raw) && length(lab$family_raw) == vcount(g))

# families numbered by size in posts (1 = largest); isolated topics form one-topic families of their own
fam_tab <- lab |> group_by(family_raw) |>
  summarise(n_topics = n(), n_posts = sum(n_cluster), .groups = "drop") |>
  arrange(desc(n_posts)) |> mutate(family = row_number())
lab <- lab |> left_join(fam_tab |> select(family_raw, family), by = "family_raw") |> select(-family_raw)

set.seed(SEED)
cmp <- components(g)
conn <- cmp$membership == which.max(cmp$csize)      # the main connected network
gm <- induced_subgraph(g, which(conn))
xy <- layout_with_fr(gm, weights = E(gm)$weight, niter = 1000)
lab$x <- NA_real_; lab$y <- NA_real_
lab$x[conn] <- xy[, 1]; lab$y[conn] <- xy[, 2]
lab$in_main_network <- conn
log_msg("families: %d (%d with more than one topic); isolated topics: %d; main network: %d topics (%d outside it)",
        nrow(fam_tab), sum(fam_tab$n_topics > 1), sum(lab$degree == 0), sum(conn), sum(!conn))

# ---------------------------------------------------------------------------
# 4. Tables
# ---------------------------------------------------------------------------
families <- lab |> group_by(family) |>
  arrange(desc(n_cluster), .by_group = TRUE) |>
  summarise(n_topics = n(), n_posts = sum(n_cluster),
            bins = paste(sort(unique(period_bin)), collapse = " "), n_bins = n_distinct(period_bin),
            share_topics_wc_bins = round(mean(BIN_TYPE[period_bin] == "tournament"), 3),
            largest_topics = paste(head(name, 5), collapse = " | "),
            label = paste(head(unique(name), 2), collapse = " / "),   # two largest distinct names (two families may share the largest)
            top_tones = paste(head(names(sort(table(tone_category), decreasing = TRUE)), 3), collapse = ", "),
            .groups = "drop")
save_tab(families, "D1_topic_families")
save_tab(lab |> select(topic_uid, period_bin, topic_id, name, tone_category, n_cluster, family, degree, in_main_network, x, y), "D1_topic_network_nodes")
save_tab(edges |> select(from, to, similarity, same_bin) |> mutate(similarity = round(similarity, 4)), "D1_topic_network_edges")

# ---------------------------------------------------------------------------
# 5. Figures (the same layout twice)
# ---------------------------------------------------------------------------
pts <- lab |> filter(conn)
seg <- edges |> filter(conn[a], conn[b]) |> transmute(x = lab$x[a], y = lab$y[a], xend = lab$x[b], yend = lab$y[b])   # links of the drawn network
top <- families |> filter(family <= N_COLOUR) |>
  mutate(key = sprintf("F%d", family),
         legend = sprintf("F%d  %s  (%d topics, bins %s)", family, label, n_topics,
                          ifelse(n_bins == 9, "1–9", bins)))
pts <- pts |> mutate(colour_key = ifelse(family <= N_COLOUR, sprintf("F%d", family), "other"))
centres <- pts |> filter(family <= N_COLOUR) |> group_by(family) |>
  summarise(x = median(x), y = median(y), .groups = "drop") |> mutate(key = sprintf("F%d", family))
n_other <- nrow(families) - N_COLOUR
lim_x <- quantile(pts$x, c(0.01, 0.99)); lim_y <- quantile(pts$y, c(0.01, 0.99))
n_out <- sum(pts$x < lim_x[1] | pts$x > lim_x[2] | pts$y < lim_y[1] | pts$y > lim_y[2])
cap <- sprintf(paste0("Each dot is one of %s topics of the nine period bins (size = posts in its HDBSCAN cluster). A line joins a topic to each of its %d most similar topics anywhere in\n",
                      "the corpus (cosine similarity of the topic embeddings, all-mpnet-base-v2, at least %.2f): %s links, %.0f %% of them between different bins. Families: Louvain communities; each\n",
                      "is named here after its two largest topics. Not drawn: %d topics outside the main network (isolated, or in small groups linked only to each other); %d topics of the\n",
                      "main network lie beyond the frame (central 98 %% of positions)."),
               format(nrow(lab), big.mark = ","), K, MIN_SIM, format(nrow(edges), big.mark = ","), 100 * mean(!edges$same_bin), sum(!conn), n_out)
net_theme <- theme_fifa() + theme(axis.text = element_blank(), axis.title = element_blank(), axis.line.x = element_blank(),
                                   panel.grid.major.y = element_blank(), legend.position = "right", legend.justification = "top",
                                   legend.key.height = unit(0.9, "lines"))
size_scale <- scale_size_area(max_size = 3.2, guide = "none")
base <- ggplot() + geom_segment(data = seg, aes(x, y, xend = xend, yend = yend), colour = GRID, linewidth = 0.1, alpha = 0.6) +
  coord_equal(xlim = lim_x, ylim = lim_y) + net_theme

family_map <- function(labelled, bold = FALSE) {
  # bold (D1b, author's request): every topic gets a visible minimum size and the coloured families are drawn larger
  # than the grey background, so the clusters read without labels
  grow <- if (bold) 1.8 else 1
  p <- base +
    geom_point(data = filter(pts, colour_key == "other"), aes(x, y, size = n_cluster), colour = AXIS, alpha = if (bold) 0.5 else 0.7, stroke = 0) +
    geom_point(data = filter(pts, colour_key != "other"), aes(x, y, size = n_cluster * grow, colour = colour_key), alpha = if (bold) 0.95 else 0.85, stroke = 0)
  if (labelled)
    p <- p + geom_label_repel(data = centres, aes(x, y, label = key), size = 2.8, colour = INK, fill = SURFACE, label.size = 0.2,
                              min.segment.length = 0, seed = SEED, box.padding = 0.4)
  p + scale_colour_manual(values = setNames(SERIES, top$key), breaks = top$key, labels = top$legend, name = "Largest families") +
    (if (bold) scale_size(range = c(0.6, 4.5), guide = "none") else size_scale) +
    guides(colour = guide_legend(override.aes = list(size = 2.5, alpha = 1))) +
    labs(title = if (bold) "Semantic similarity of r/soccer post-topic clusters, 2010–2026" else "How the topics of r/soccer relate to one another, 2010–2026",
         subtitle = if (bold) NULL else          # D1b: the author's title, no subtitle
           sprintf("Topics linked by semantic similarity across all nine period bins; coloured: the eight largest families by posts, grey: the other %s", format(n_other, big.mark = ",")),
         caption = cap)
}
save_fig(family_map(TRUE), "D1_topic_network_families", w = 14, h = 8.5)
save_fig(family_map(FALSE, bold = TRUE), "D1b_topic_network_families_unlabelled", w = 14, h = 8.5)   # author's request: dots and links only

# D2: one panel per period bin — the bin's topics highlighted on the whole network (grey); nine shades of one hue
# cannot be told apart, and 84 % of the links cross bins, so the bins are shown side by side instead
bin_lab <- sprintf("Bin %d  %s – %s%s", 1:9, format(BIN_START, "%b %Y"), format(BIN_END, "%b %Y"), ifelse(BIN_TYPE == "tournament", "  (World Cup)", ""))
hi <- pts |> mutate(panel = factor(bin_lab[period_bin], bin_lab))
bg <- pts |> select(x, y, n_cluster) |> tidyr::crossing(panel = factor(bin_lab, bin_lab))
p <- ggplot() +
  geom_point(data = bg, aes(x, y, size = n_cluster), colour = GRID, stroke = 0) +
  geom_point(data = hi, aes(x, y, size = n_cluster, colour = BIN_TYPE[period_bin]), alpha = 0.85, stroke = 0) +
  facet_wrap(~ panel, ncol = 3) +
  scale_colour_manual(values = c(tournament = SERIES[1], interim = SERIES[2]), breaks = c("tournament", "interim"),
                      labels = c(tournament = "World Cup period bin", interim = "Interim period bin")) +
  scale_size(range = c(0.5, 2.6), guide = "none") + coord_equal(xlim = lim_x, ylim = lim_y) + net_theme +   # small panels: a visible floor
  theme(legend.position = "top", strip.text = element_text(size = 8)) +
  guides(colour = guide_legend(override.aes = list(size = 2.5))) +
  labs(title = "Where each period's topics sit on the topic map",
       subtitle = "Same layout as the family map; each panel highlights the topics of one period bin over all 6,053 topics of the main network (grey)",
       caption = cap)
save_fig(p, "D2_topic_network_periods", w = 10, h = 11)

if (!all(unlist(results))) stop("a check failed")
log_msg("written to %s", OUT_DIR)
