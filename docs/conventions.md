# App-wide conventions

Cross-tab mechanisms: ship lags, section deep links, today-forward mode, and where caveats go.

## Trained-to-release lags

`us_ship_lag` (70 days) and `cn_ship_lag` (21), `_SHIP_LAG_DEFAULTS`, are app-wide sidebar
settings (`_ship_lag_inputs`: US on Takeoff, both on Pacing), read by `_ship_lag` and
reassigned at dispatch so a tab switch keeps them. Takeoff and Pacing date *trained* models.
Takeoff's ECI paths start at the latest US release plus the US lag of the fitted climb
(`_tk_eci_scale(lead_years=…)`; `eci["released"]` keeps the release for the yearly chart;
`test_today_is_the_trained_frontier_a_ship_lag_past_the_release`), and release-dated
milestone cards are pulled back over 0.7–1.3× the US lag (`_pc_report_lag_days`). On Pacing
a China release at t races the US as of t less China's lag (`us_path_cn`); the bar, the
milestone table and the time to surpass are trained against trained on every clock, while
on *Model release* each side's dates add its own lag (the race's too, by site country), so
China's sit 49 days nearer the US's (`test_pause_dates_follow_timing_in_lockstep`,
`test_a_longer_us_release_lag_raises_the_trained_bar`). The Data Centers tab's *Model
release* does the same per site (`_dc_release_lag`: China's lag for mainland and Johor sites
and China's country lines), shows the settings only on that clock, and its hovers' *Model
out* line uses the site's lag (`test_release_dates_follow_each_countrys_lag`).

## Section deep links

`?to=<heading anchor>` scrolls to a section; a bare `#fragment` cannot. Community Cloud
serves the app in an iframe whose src copies the query string and drops the fragment, and
every `st.query_params` write rebuilds the URL from path + search alone, dropping it again.
Three things are load-bearing:

1. **The slug is whitelisted, not escaped** (`_anchor_slug()`): it is inlined into the
   injected script as a bare literal.
2. **`to` is consumed on arrival.** Left in the URL, every later rerun would jump the page
   back, and *Share view* would hand out the old section.
3. **`_render_anchor_links()` rewrites each main-column heading's link icon** to an absolute
   `?tab=…&to=…` URL of the outer page, and intercepts the click as an in-page scroll —
   Streamlit's own icon offers the `#fragment` form. Sidebar headings are left alone. The
   arrival scroll polls for the heading (the page streams in) and re-settles briefly (Plotly
   resizes after insert), stopping on the reader's first scroll.

`TestAnchorSlug` and `TestSectionDeepLinks` guard it; the latter also holds every heading
inside the whitelist, since the icon hands out a link the app must then accept. The METR tab
has no main-column headings, so it has no section to link to.

## Today-forward mode

A sidebar checkbox (`today_fwd`, URL-tracked, reset by no tab) at the foot of
the sidebar on the six projection tabs (`_TODAY_FWD_TABS` — METR, ECI,
RLI, Takeoff, Revenue, Pacing), rendered by the dispatch after the tab's own
controls. On, a chart opens at today with no history and every hover date
reads relative to today (`_rel_date_label`: days to 30, then the two largest
of y/mo/d, calendar months). It is a post-process on the finished figure,
`_today_forward()`, applied by wrapping the chart in `_tf()` at
`st.plotly_chart`; nothing about the tab's hover builders changes. Four things
are load-bearing:

1. **Only the projection charts opt in** — the RSI tab's CoBench, survey, code
   and detour charts are diagnostic and keep their history; the blend's CDF is
   the one RSI chart wrapped. Tables, cards and annotations keep calendar
   dates; the toggle's help says so.
2. **A `hovertemplate`'s `%{x|…}` becomes a `customdata` column** of relative
   labels, since Plotly can only format x as a calendar date; a trace that
   already carries `customdata` is left alone. `text` is rewritten only when
   it serves as hover text and is not drawn on the chart.
3. **Unified hover falls back to per-point hover**, because the unified header
   prints the x value as a calendar date and nothing can reformat it.
4. **The y-axis refits to the forward data** (`_today_fwd_refit_y`) when that
   data covers under `_TODAY_FWD_Y_COVER` of the recorded range — an axis sized
   to its history leaves the fan in a sliver at the top — and is kept when the
   curve mostly spans it (the blend's 0–100% CDF). Two-point hover-skipped
   traces are reference levels, not data; a log axis refits in log10, a
   categorical one (the Pacing race timeline) is left alone. The legend moves
   above the plot, since the fan now starts under an inside top-left one.

Guarded by `TestTodayForward` (unit) and `TestTodayForwardMode` (integration).

## Caveats belong on the thing they qualify

Prefer a hover to a paragraph — applied across every tab. Keep in the visible
line only what a reader needs *before* deciding to look closer: the claim the
section makes, and anything the page would be misleading without. The method, the
provenance and the caveats go in hovers. Chart **legends stay visible** — they are
keys, not caveats.

Three mechanisms, in order of preference:

1. **`st.metric(..., help=…)`** where a metric exists. Native tooltip, no raw
   HTML. The *Capabilities Milestones* row is the worked example: ten cards,
   each with its own note (`_notes` by slug, plus `_pc_clock_note()` for the
   release-vs-internal split), and a two-line caption under them. Widget `help=` does the same
   for controls, which is why every sidebar control has one.
2. **`_fn_caption(text, *notes)`** for prose qualifying a chart, where there is no
   widget to hang `help=` on. Each note is a `(phrase, note)` pair, and **the
   phrase is looked up in the visible line and made hoverable in place** — a
   footnote hangs off the words it qualifies, not off a marker parked at the end
   of the sentence. So write the line to contain the phrase; a phrase that isn't
   there falls back to a trailing `?`, which is the exception, not the design
   (`test_footnotes_anchor_to_their_phrase` holds the live app at zero
   fallbacks). Matching is whole-word, since a substring match would wrap "check" inside
   "Cross-checked" and split the word across a tag. It stays an `st.caption`, not
   an `st.markdown` — the element type is what tells fine print from body text, to
   the reader and to the tests that address one and not the other.
3. **`_fn_line(...)`** for a body-prominent line that must not read as fine print
   (the buildout headline). `st.warning` takes no
   HTML, so the efficiency caveats use `_fn_line` with a warning glyph instead of
   a yellow box.

`_fn_span()` wraps the anchored phrase (dotted underline, `.vp-fn-a`); `_fn()` is
the standalone fallback marker (`.vp-fn`). Their CSS lives in the global `<style>`
block at the top of the file. The bubble is dark in both themes on purpose — it
floats above the page, and no Streamlit variable reliably tracks the user's
theme.

`test_every_milestone_card_carries_its_caveats_on_hover` pins the split for the
milestone row: every card has a hover naming its fit and its clock, and the
caption stays short.

**Editing these blocks by script:** a naive paren-counter will not do — an
f-string like `.split(' (')` puts an unbalanced paren inside a literal, and a
scan that counts it runs on through thousands of lines. Tokenize, or edit by hand.
