##############################################################################
# FIFA COMMUNITIES CHAPTER
## code/R/plot-utils.R — the visual system shared by the figure scripts (11-descriptives.R, 12-topic-network.R)
##############################################################################
#
# The dataviz reference palette in light mode (print), the ggplot theme, and the writers of a figure (PNG via
# ragg at 300 dpi + PDF via cairo) and of the table behind it.  save_fig() and save_tab() write to OUT_DIR,
# which the calling script defines.  Moved out of 11-descriptives.R on 2026-10-27 when the network figure
# (12) was added, so that every figure of the chapter uses one palette and one theme.
##############################################################################

suppressPackageStartupMessages({ library(ggplot2); library(readr) })

# ---------------------------------------------------------------------------
# Visual system: the dataviz reference palette (light mode, print), thin marks, recessive chrome
# ---------------------------------------------------------------------------
INK <- "#0b0b0b"; INK2 <- "#52514e"; MUTED <- "#898781"; GRID <- "#e1e0d9"; AXIS <- "#c3c2b7"
SURFACE <- "#fcfcfb"; SHADE <- "#f0efec"
SERIES <- c("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")   # fixed order
POS <- "#2a78d6"; NEG <- "#e34948"; NEU <- MUTED                                                       # diverging pair + gray
SEQ <- c("#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")                              # one-hue blue ramp

theme_fifa <- function() {
  theme_minimal(base_size = 10, base_family = "sans") +
    theme(plot.background = element_rect(fill = SURFACE, colour = NA),
          panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.y = element_line(colour = GRID, linewidth = 0.3),
          axis.line.x = element_line(colour = AXIS, linewidth = 0.4),
          axis.text = element_text(colour = MUTED, size = 8), axis.title = element_text(colour = INK2, size = 9),
          plot.title = element_text(colour = INK, size = 12, face = "bold"),
          plot.subtitle = element_text(colour = INK2, size = 9), plot.caption = element_text(colour = MUTED, size = 7, hjust = 0),
          plot.title.position = "plot", plot.caption.position = "plot",
          strip.text = element_text(colour = INK2, size = 9, face = "bold", hjust = 0),
          legend.position = "top", legend.justification = "left", legend.text = element_text(colour = INK2, size = 8),
          legend.title = element_blank(), plot.margin = margin(10, 14, 8, 10))
}

save_fig <- function(p, id, w = 8, h = 4.5) {
  ggsave(file.path(OUT_DIR, paste0("fig_", id, ".png")), p, width = w, height = h, dpi = 300, device = ragg::agg_png, bg = SURFACE)
  ggsave(file.path(OUT_DIR, paste0("fig_", id, ".pdf")), p, width = w, height = h, device = grDevices::cairo_pdf, bg = SURFACE)   # cairo: en dashes and minus signs
  log_msg("figure %s", id)
}
save_tab <- function(t, id) write_csv(t, file.path(OUT_DIR, paste0("tab_", id, ".csv")), na = "")

