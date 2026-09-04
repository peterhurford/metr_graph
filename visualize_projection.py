"""
AI Capability Projections: Interactive Plotly fan charts for METR Horizon and Epoch ECI benchmarks.
Run: streamlit run visualize_projection.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml
import csv
import re
import html
import textwrap
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Capability Projections", layout="wide")

PROJ_DISCLAIMER = " These are projections assuming current progress continues, they are not forecasts - forecasts would involve some chance of larger-scale trend changes."

# Trajectories drawn per fan chart. Every tab samples this many paths, and that
# sampling is most of what a render costs — so the test suite turns it down via
# _VP_SAMPLES. Nothing the tests assert on depends on the count (the CI defaults
# they check come from the OLS fits, which are deterministic), and the app is
# already unseeded, so a smaller sample is only a noisier version of the same
# picture. Don't lower the default: the band edges are percentiles and get
# visibly ragged well before 5000.
N_SAMPLES = int(os.environ.get("_VP_SAMPLES", "5000"))

# Tighten default Streamlit padding, and style the inline hover footnotes
# `_fn()` emits (see "Caveats belong on the thing they qualify" in CLAUDE.md).
# The bubble is dark in both themes on purpose: it floats above the page, so
# one colour reads correctly against either background and there is no
# Streamlit variable that tracks the user's theme reliably.
st.markdown("""<style>
    .block-container { padding-top: 2rem !important; }
    [data-testid="stTable"] table { margin-top: 0 !important; margin-bottom: 0.5rem !important; }
    .vp-fn, .vp-fn-a { position: relative; cursor: help; }
    /* Anchored: the note hangs off the words it qualifies. */
    .vp-fn-a {
        display: inline;
        border-bottom: 1px dotted currentColor;
        text-decoration-skip-ink: none;
    }
    .vp-fn-a:hover { border-bottom-style: solid; }
    /* Fallback marker, for a note with no phrase to sit on. */
    .vp-fn {
        display: inline-block;
        font-size: 0.85em; font-weight: 600; line-height: 1;
        padding: 0 0.28em; margin: 0 0.1em; border-radius: 0.65em;
        border: 1px solid currentColor; opacity: 0.75; vertical-align: baseline;
    }
    .vp-fn:hover { opacity: 1; }
    .vp-fn > .vp-fn-body, .vp-fn-a > .vp-fn-body {
        visibility: hidden; opacity: 0; transition: opacity 0.12s ease;
        position: absolute; bottom: 1.6em; left: 0; z-index: 1000;
        width: max-content; max-width: min(30rem, 70vw);
        padding: 0.55rem 0.7rem; border-radius: 0.45rem;
        background: rgba(28, 29, 36, 0.97); color: #F4F4F6;
        border: 1px solid rgba(255, 255, 255, 0.14);
        box-shadow: 0 3px 14px rgba(0, 0, 0, 0.32);
        font-size: 0.82rem; font-weight: 400; line-height: 1.42;
        white-space: normal; text-align: left; opacity: 0;
    }
    .vp-fn:hover > .vp-fn-body,
    .vp-fn-a:hover > .vp-fn-body { visibility: visible; opacity: 1; }
    /* Near the right edge the bubble would overflow the column. */
    .vp-fn-r > .vp-fn-body { left: auto; right: 0; }
</style>""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────

def _fn(note, marker="?", right=False):
    """A standalone hover-footnote marker, for a note with no phrase to sit on.

    Prefer anchoring — `_fn_caption()` wraps a note's marker where it occurs
    in the visible line, so the footnote hangs off the words it qualifies
    rather than trailing the sentence. This is the fallback for the rare
    note whose subject isn't nameable in the text.
    """
    return _fn_span(marker, note, cls="vp-fn", right=right)


def _fn_span(text, note, cls="vp-fn-a", right=False):
    """`text`, hoverable, with `note` in the bubble."""
    body = html.escape(str(note).strip()).replace("\n", " ")
    if right:
        cls += " vp-fn-r"
    return (f'<span class="{cls}">{text}'
            f'<span class="vp-fn-body">{body}</span></span>')


def _fn_anchor(text, notes, right=None):
    """Attach each note to its own phrase in `text`.

    A note is a `(phrase, note)` pair; the phrase is looked up in the visible
    line and wrapped in place, so the dotted underline sits on the words the
    note is about. A phrase that isn't in the line (or a bare string note)
    falls back to a trailing `?` marker — the anchored form is the default
    and the trailing one is the exception, not the other way round.

    Occurrences are spliced back-to-front so earlier offsets stay valid, and
    a phrase that would land inside an already-wrapped span is skipped
    rather than nesting.
    """
    def _find(phrase):
        """First whole-word occurrence, or -1.

        Word-bounded on purpose: a bare `find` once wrapped "check" inside
        "Cross-checked", splitting the word across a tag.
        """
        for m in re.finditer(re.escape(phrase), text):
            a, b = m.start(), m.end()
            if (a and (text[a - 1].isalnum() or text[a - 1] == '_')):
                continue
            if b < len(text) and (text[b].isalnum() or text[b] == '_'):
                continue
            return a
        return -1

    spans, trailing = [], []
    for n in notes:
        phrase, note = n if isinstance(n, (tuple, list)) else (None, n)
        i = _find(phrase) if phrase else -1
        if i < 0 or any(i < e and s_ < i + len(phrase) for s_, e, _ in spans):
            trailing.append((phrase, note))
            continue
        spans.append((i, i + len(phrase), note))
    out = text
    # Right-anchor a bubble whose phrase starts past ~⅔ of the line, where a
    # left-anchored one would run off the column.
    cut = len(text) * 2 // 3
    for start, end, note in sorted(spans, reverse=True):
        out = (out[:start]
               + _fn_span(out[start:end], note,
                          right=(start > cut) if right is None else right)
               + out[end:])
    return out + "".join(
        _fn(note, phrase or "?", right=True) for phrase, note in trailing)


def _fn_caption(text, *notes, right=None):
    """A caption whose footnotes hang off the phrases they qualify.

    Stays an `st.caption` rather than an `st.markdown` — the element type is
    what tells fine print from body text, both to the reader and to the
    tests that address one and not the other.
    """
    st.caption(_fn_anchor(text, notes, right), unsafe_allow_html=True)


def _fn_line(text, *notes, right=None, parent=None):
    """Body-prominent prose with anchored footnotes, for a line that must not
    read as fine print (the pause panel's *Assumes*)."""
    (parent or st).markdown(_fn_anchor(text, notes, right),
                            unsafe_allow_html=True)


def _fmt_jobs(m):
    """Format a jobs number (in millions) readably: <1M shows as K, ≥1M shows as M."""
    if abs(m) < 1.0:
        return f"{m * 1000:.0f}K"
    return f"{m:.1f}M"

_NAMES = {
    'gpt2': 'GPT-2', 'davinci_002': 'davinci-002',
    'gpt_3_5_turbo_instruct': 'GPT-3.5T', 'gpt_4': 'GPT-4',
    'gpt_4_1106_inspect': 'GPT-4 Nov23', 'gpt_4o_inspect': 'GPT-4o',
    'claude_3_5_sonnet_20240620_inspect': 'Claude 3.5S (old)',
    'o1_preview': 'o1-pre', 'claude_3_5_sonnet_20241022_inspect': 'Claude 3.5S (new)',
    'o1_inspect': 'o1', 'claude_3_7_sonnet_inspect': 'Claude 3.7S',
    'o3_inspect': 'o3', 'gpt_5_2025_08_07_inspect': 'GPT-5',
    'gemini_3_pro': 'Gemini 3 Pro',
    'gpt_5_1_codex_max_inspect': 'GPT-5.1 Codex',
    'claude_opus_4_5_inspect': 'Claude 4.5 Opus', 'gpt_5_2': 'GPT-5.2',
    'claude_3_opus_inspect': 'Claude 3 Opus', 'gpt_4_turbo_inspect': 'GPT-4T',
    'claude_4_opus_inspect': 'Claude 4 Opus',
    'claude_opus_4_6_inspect': 'Claude 4.6 Opus',
    'gpt_5_3_codex': 'GPT-5.3 Codex',
    'gpt_5_4_xhigh': 'GPT-5.4 (xhigh)',
    'claude_mythos_preview_early_inspect': 'Claude Mythos',
}


def pretty(name):
    return _NAMES.get(name, name)


def log2min_to_label(val, hours_only=False):
    """Convert log2(minutes) to human-readable string."""
    minutes = 2 ** val
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    if minutes < 60:
        return f"{minutes:.0f}m"
    hrs = minutes / 60
    return fmt_hrs(hrs, hours_only=hours_only)


def fmt_hrs(h, hours_only=False):
    """Format hours for display using work-time units (8h/d, 40h/w, 176h/mo, 2000h/y).
    Shows sub-unit remainder (e.g., 1h20m, 2d3h). No decimals.
    When hours_only=True, output is always in hours (or minutes for h<1)."""
    minutes = h * 60
    if hours_only:
        if h < 1:
            return f"{int(round(minutes))}m"
        return f"{int(round(h)):,}h"
    if h < 1:
        return f"{int(round(minutes))}m"
    if h < 100:
        hrs = int(h)
        mins = int(round((h - hrs) * 60))
        if mins == 60:
            hrs += 1
            mins = 0
        if mins == 0:
            return f"{hrs}h"
        return f"{hrs}h{mins}m"
    days_ = h / 8
    if days_ < 5:
        d = int(days_)
        rem_h = int(round(h - d * 8))
        if rem_h == 8:
            d += 1
            rem_h = 0
        if rem_h == 0:
            return f"{d}d"
        return f"{d}d{rem_h}h"
    weeks = h / 40
    if weeks < 4.4:
        w = int(weeks)
        rem_d = int(round((h - w * 40) / 8))
        if rem_d == 5:
            w += 1
            rem_d = 0
        if rem_d == 0:
            return f"{w}w"
        return f"{w}w{rem_d}d"
    months = h / 176
    if h < 2000:
        mo = int(months)
        rem_w = int(round((h - mo * 176) / 40))
        if rem_w == 4:
            mo += 1
            rem_w = 0
        if rem_w == 0:
            return f"{mo}mo"
        return f"{mo}mo{rem_w}w"
    years = h / 2000
    y = int(years)
    rem_mo = int(round((h - y * 2000) / 176))
    if rem_mo == 11:
        y += 1
        rem_mo = 0
    if rem_mo == 0:
        return f"{y}y"
    return f"{y}y{rem_mo}mo"


def fit_line(x, y):
    A = np.column_stack([np.ones_like(x), x])
    params, *_ = np.linalg.lstsq(A, y, rcond=None)
    return params


def _fit_slope_p50_intercept_display(d, p50_y, disp_y):
    """Fit slope on p50, then compute best intercept for display data with that slope."""
    params_p50 = fit_line(d, p50_y)
    slope = params_p50[1]
    intercept = np.mean(disp_y - slope * d)
    return np.array([intercept, slope])


def _lognormal_from_ci(lo, hi, n):
    """Sample from lognormal fitted to 80% CI bounds [lo, hi]."""
    mu_ln = (np.log(lo) + np.log(hi)) / 2
    sigma_ln = (np.log(hi) - np.log(lo)) / (2 * 1.282)
    return np.random.lognormal(mu_ln, sigma_ln, n)


def _normal_from_ci(lo, hi, n):
    """Sample from normal fitted to 80% CI bounds [lo, hi], clipped at lo/10."""
    mu = (lo + hi) / 2
    sigma = (hi - lo) / (2 * 1.282)
    return np.maximum(np.random.normal(mu, sigma, n), lo / 10)


def _log_lognormal_from_ci(lo, hi, n):
    """Sample from log-lognormal fitted to 80% CI bounds [lo, hi].

    log(X) ~ Lognormal  (i.e. log(log(X)) ~ Normal).
    Gives a much fatter right tail than lognormal: the distribution is
    right-skewed even in log-space.  Requires lo > 1 and hi > 1.
    """
    log_lo, log_hi = np.log(lo), np.log(hi)
    mu_y = (np.log(log_lo) + np.log(log_hi)) / 2
    sigma_y = (np.log(log_hi) - np.log(log_lo)) / (2 * 1.282)
    log_x = np.random.lognormal(mu_y, max(sigma_y, 0), n)
    return np.exp(log_x)


def _ss_number_input(parent, label, key, default, **kwargs):
    """number_input with session_state-driven default.

    Avoids Streamlit's conflict between value= and Session State API on reset:
    the widget has no explicit value= so there's never a clash, and session_state
    is initialised to *default* on first render (or after a reset pops the key).
    """
    if key not in st.session_state:
        st.session_state[key] = default
    return parent.number_input(label, key=key, **kwargs)


def superexp_trajectory(days, dt_0, halflife, dt_floor):
    """Deterministic superexponential trajectory starting at 0.

    Returns the cumulative growth (in the same units as the y-axis) over `days`.
    DT decays exponentially: dt(t) = dt_0 * 2^(-t/halflife), floored at dt_floor.
    Growth = integral of 1/dt(t) dt, so:
      - Before floor hit: (H / (dt_0 * ln2)) * (2^(t/H) - 1)
      - After floor hit:  linear at rate 1/dt_floor

    dt_0 can be a 1-D array of shape (n_samples,); in that case days should be
    1-D (n_days,) and the result is (n_samples, n_days).
    """
    dt_0 = np.asarray(dt_0)
    if dt_0.ndim == 0:
        # scalar path (unchanged)
        t_cap = halflife * np.log2(dt_0 / dt_floor) if dt_0 > dt_floor else 0.0
        se_phase = np.minimum(days, t_cap)
        y_se = (halflife / (dt_0 * np.log(2))) * (2 ** (se_phase / halflife) - 1)
        linear_phase = np.maximum(days - t_cap, 0)
        y_lin = linear_phase / dt_floor
        return y_se + y_lin
    # vectorised path: dt_0 is (n_samples,), days is (n_days,)
    dt_0 = dt_0[:, None]                                 # (n, 1)
    days = np.asarray(days)[None, :]                      # (1, d)
    t_cap = np.where(dt_0 > dt_floor,
                     halflife * np.log2(dt_0 / dt_floor), 0.0)  # (n, 1)
    se_phase = np.minimum(days, t_cap)                    # (n, d)
    y_se = (halflife / (dt_0 * np.log(2))) * (2 ** (se_phase / halflife) - 1)
    linear_phase = np.maximum(days - t_cap, 0)            # (n, d)
    y_lin = linear_phase / dt_floor
    return y_se + y_lin                                   # (n, d)


def _logit(p):
    """Logit transform: log(p / (1-p)). p in (0,1)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.log(p / (1 - p))


def _inv_logit(x):
    """Inverse logit (sigmoid): 1 / (1 + exp(-x))."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ── Backtesting helpers ──────────────────────────────────────────────────

def _backtest_stats(future_models, all_trajectories, proj_start_date, proj_end_date,
                    get_value, get_name):
    """Compute backtest statistics for future frontier models vs. projected trajectories."""
    results = []
    for m in future_models:
        if m['date'] <= proj_start_date or m['date'] > proj_end_date:
            continue
        day_idx = (m['date'] - proj_start_date).days
        if day_idx < 0 or day_idx >= all_trajectories.shape[1]:
            continue
        traj_col = all_trajectories[:, day_idx]
        val = get_value(m)
        pctile = float(np.mean(traj_col <= val) * 100)
        p5, p10, p25, p75, p90, p95 = np.percentile(traj_col, [5, 10, 25, 75, 90, 95])
        results.append({
            'model': m, 'name': get_name(m), 'date': m['date'], 'value': val,
            'percentile': pctile,
            'within_50': bool(p25 <= val <= p75),
            'within_80': bool(p10 <= val <= p90),
            'within_90': bool(p5 <= val <= p95),
        })
    return results


def _bt_color_for(r):
    """Return color for a backtest result based on CI band membership."""
    if r['within_50']:
        return '#27ae60'
    if r['within_80']:
        return '#f1c40f'
    if r['within_90']:
        return '#e67e22'
    return '#e74c3c'


def _add_today_vline(fig):
    """The dashed 'Today' divider every projection chart carries."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'),
                  opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')
    return today


def _add_backtest_traces(fig, backtest_results, proj_start_date, yconv=None):
    """Add cutoff line and actual trajectory line to a plotly figure."""
    _yc = yconv if yconv else (lambda x: x)
    # Cutoff line
    fig.add_vline(
        x=proj_start_date,
        line=dict(color='#e67e22', width=2, dash='dash'),
        opacity=0.8,
    )
    fig.add_annotation(
        x=proj_start_date, y=1.0, yref='paper',
        text='  Projection start', showarrow=False, textangle=-90,
        font=dict(size=10, color='#e67e22'),
        xanchor='right', yanchor='top',
    )
    # Actual trajectory line
    if len(backtest_results) >= 2:
        dates = [r['date'] for r in backtest_results]
        values = [_yc(r['value']) for r in backtest_results]
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode='lines',
            line=dict(color='#27ae60', width=2, dash='dash'),
            name='Actual trajectory',
            hoverinfo='skip', showlegend=True,
        ))


def _backtest_summary(backtest_results):
    """Show st.info() summary bar for backtest results."""
    if not backtest_results:
        return
    n = len(backtest_results)
    n_50 = sum(1 for r in backtest_results if r['within_50'])
    n_80 = sum(1 for r in backtest_results if r['within_80'])
    n_90 = sum(1 for r in backtest_results if r['within_90'])
    mean_pct = np.mean([r['percentile'] for r in backtest_results])
    st.info(
        f"**Backtest: {n} future models.** "
        f"Within 50% CI: {n_50}/{n} | "
        f"Within 80% CI: {n_80}/{n} | "
        f"Within 90% CI: {n_90}/{n} | "
        f"Mean percentile: {mean_pct:.0f}%"
    )


# ── Data loading ─────────────────────────────────────────────────────────

def _yaml_mtime():
    yaml_path = os.path.join(os.path.dirname(__file__), 'benchmark_results_1_1.yaml')
    return os.path.getmtime(yaml_path)


@st.cache_data
def load_metr_all(_mtime=None):
    yaml_path = os.path.join(os.path.dirname(__file__), 'benchmark_results_1_1.yaml')
    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)

    models = []
    for key, result in raw['results'].items():
        metrics = result['metrics']
        p50 = metrics.get('p50_horizon_length', {}).get('estimate')
        p50_lo = metrics.get('p50_horizon_length', {}).get('ci_low')
        p50_hi = metrics.get('p50_horizon_length', {}).get('ci_high')
        p80 = metrics.get('p80_horizon_length', {}).get('estimate')
        p80_lo = metrics.get('p80_horizon_length', {}).get('ci_low')
        p80_hi = metrics.get('p80_horizon_length', {}).get('ci_high')
        is_sota = metrics.get('is_sota', False)
        if p50 is not None:
            rd = result['release_date']
            if isinstance(rd, str):
                rd = datetime.strptime(rd, '%Y-%m-%d')
            else:
                rd = datetime(rd.year, rd.month, rd.day)
            models.append({
                'name': key, 'date': rd,
                'p50_min': p50, 'p50_lo': p50_lo, 'p50_hi': p50_hi,
                'p80_min': p80, 'p80_lo': p80_lo, 'p80_hi': p80_hi,
                'is_sota': is_sota,
            })

    models.sort(key=lambda m: m['date'])
    return models


@st.cache_data
def load_frontier(_mtime=None):
    return [m for m in load_metr_all(_mtime=_mtime) if m['is_sota']]


def _eci_mtime():
    csv_path = os.path.join(os.path.dirname(__file__), 'epoch_capabilities_index.csv')
    return os.path.getmtime(csv_path)


@st.cache_data
def load_eci_frontier(_mtime=None, country=None, orgs=None, full_window=False):
    csv_path = os.path.join(os.path.dirname(__file__), 'epoch_capabilities_index.csv')
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    valid = []
    for r in rows:
        score_str = r.get('ECI Score', '').strip()
        date_str = r.get('Release date', '').strip()
        if not score_str or not date_str:
            continue
        try:
            score = float(score_str)
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        valid.append({
            'version': r.get('Model version', ''),
            'name': r.get('Model name', ''),
            'display_name': (r.get('Display name', '') or '').strip() or r.get('Model name', ''),
            'date': date,
            'eci_score': score,
            'organization': r.get('Organization', ''),
            'country': r.get('Country', ''),
        })

    # Dedup: keep highest-scoring variant per model name
    best_by_name = {}
    for m in valid:
        name = m['name']
        if name not in best_by_name or m['eci_score'] > best_by_name[name]['eci_score']:
            best_by_name[name] = m
    deduped = sorted(best_by_name.values(), key=lambda m: m['date'])

    # Filter to Claude 3 Opus era onward (Feb 2024+). full_window=True skips
    # the cutoff — required wherever the running frontier is compared against
    # earlier models (the frontier-grade refit), where the cutoff would leave
    # them with no frontier at all.
    if not full_window:
        _cutoff_date = datetime(2024, 2, 29)
        deduped = [m for m in deduped if m['date'] >= _cutoff_date]

    # Optional country filter
    if country:
        deduped = [m for m in deduped if m.get('country') == country]

    # Optional organization filter: keep models whose Organization field
    # contains any of the given substrings (case-insensitive). Handles the
    # comma-joined multi-org strings in the CSV (e.g. "Google DeepMind,Google").
    if orgs:
        _orgs_l = [o.lower() for o in orgs]
        deduped = [m for m in deduped
                   if any(o in m.get('organization', '').lower() for o in _orgs_l)]

    # Frontier detection: running max
    max_score = -float('inf')
    for m in deduped:
        if m['eci_score'] > max_score:
            max_score = m['eci_score']
            m['is_frontier'] = True
        else:
            m['is_frontier'] = False

    return deduped


@st.cache_data
def load_eci_compute(_mtime=None):
    """Models that have BOTH an ECI score and a training-compute (FLOP) figure.

    Used by the Compute/capabilities/diffusion tab to regress ECI on log10(training
    FLOP) and time. Returns a list of dicts sorted by date:
    {date, eci, log10_flop, name, organization, country, is_eci_frontier}.
    The frontier flag is the running-max ECI within this compute-having subset.
    No date cutoff — we want every model with both fields for the fit.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'epoch_capabilities_index.csv')
    with open(csv_path, 'r') as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        score_str = r.get('ECI Score', '').strip()
        flop_str = (r.get('Training compute (FLOP)', '') or '').strip().replace(',', '')
        date_str = r.get('Release date', '').strip()
        if not score_str or not flop_str or not date_str:
            continue
        try:
            score = float(score_str)
            flop = float(flop_str)
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        if flop <= 0:
            continue
        out.append({
            'date': date,
            'eci': score,
            'log10_flop': float(np.log10(flop)),
            'name': (r.get('Display name', '') or '').strip() or r.get('Model name', ''),
            'organization': r.get('Organization', ''),
            'country': r.get('Country', ''),
        })
    out.sort(key=lambda m: m['date'])
    max_score = -float('inf')
    for m in out:
        m['is_eci_frontier'] = m['eci'] > max_score
        if m['eci'] > max_score:
            max_score = m['eci']
    return out


# ── RLI data (hardcoded – small dataset from remotelabor.ai) ─────────────

_RLI_RAW = [
    {"name": "Gemini 2.5 Pro", "date": "2025-03-25", "rli_score": 0.83},
    {"name": "Grok 4",         "date": "2025-07-01", "rli_score": 2.08},
    {"name": "GPT-5",          "date": "2025-08-07", "rli_score": 1.67},
    {"name": "Sonnet 4.5",     "date": "2025-09-20", "rli_score": 2.08},
    {"name": "Manus 1.5",      "date": "2025-10-20", "rli_score": 2.50},
    {"name": "Opus 4.5",       "date": "2025-11-15", "rli_score": 3.75},
    {"name": "Gemini 3 Pro",   "date": "2025-12-10", "rli_score": 1.25},
    {"name": "Manus 1.6",      "date": "2025-12-15", "rli_score": 2.92},
    {"name": "GPT-5.2",        "date": "2025-12-20", "rli_score": 2.50},
    {"name": "Opus 4.6",       "date": "2026-02-05", "rli_score": 4.17},
    {"name": "GPT-5.5",        "date": "2026-04-23", "rli_score": 6.25},
    {"name": "Opus 4.8",       "date": "2026-05-28", "rli_score": 8.33},
    {"name": "Fable 5",        "date": "2026-06-09", "rli_score": 15.83},
    # Added 2026-08-29: Scale added Gemini 3.7 Flash on 2026-08-24 (the only row
    # on the board flagged new since our last check). Score 5.00 from the
    # leaderboard's embedded JSON payload; date is Google's announced release.
    # Well below Fable 5, so it is not a frontier point.
    {"name": "Gemini 3.7 Flash", "date": "2026-08-13", "rli_score": 5.00},
    # Rechecked 2026-08-21 against labs.scale.com/leaderboard/rli and dashboard.safe.ai:
    # no entry newer than Fable 5, no changed score on any model we carry, and the CAIS
    # blog (2026-07-01) is still the latest RLI announcement. GPT-5.6 Sol, the Gemini 3.x
    # line, Kimi K3, Grok 4.6, DeepSeek V4, Qwen 3.8, and Claude Opus 5 have no published
    # RLI score anywhere -- the dashboard.safe.ai bundle already carries model-registry
    # entries for several of them, so their absence is a missing score, not a missing model.
    # remotelabor.ai no longer hosts a table at all; it now just points at dashboard.safe.ai.
    # A "16.1%" for Fable 5 circulates via a third-party blog (Pebblous) citing "CAIS
    # (2026)" with no link; contradicted by all three primary surfaces (Scale 15.80,
    # CAIS blog 15.8, dashboard.safe.ai 15.83). Treated as an error, not a revision.
    # Two standing discrepancies, deliberately left as-is:
    #   * Scale renders Fable 5 as 15.80, the CAIS dashboard as 15.83. We keep 15.83 --
    #     the benchmark's denominator is 240 projects and 38/240 = 15.833%, so 15.80
    #     is a rounding artifact. A "16.1%" figure also circulates (secondary reporting
    #     attributes it to CAIS and to scoring 218 of 240 jobs, 22 being unscorable);
    #     that denominator is not stated on either canonical surface, so we keep the
    #     dashboard's 15.83 -- remotelabor.ai designates the dashboard as canonical.
    #   * Grok 4 has been dropped from the live leaderboard (it was in the Oct 2025
    #     paper). Its 2.08 is not contradicted, just no longer reproducible upstream.
    # Leaderboard rows below our floor and intentionally not carried: Manus 1.0 (2.50),
    # ChatGPT agent (1.25), gpt-5.2 (default) 2.08 -- we carry the (medium) variant.
    # Rechecked 2026-09-03: no new model and no changed score on any of the four surfaces.
    # Scale's embedded JSON still tops out at Gemini 3.7 Flash (createdAt 2026-08-24); every
    # other row's createdAt is 2026-03-05 or earlier. safe.ai/blog's newest post is still
    # 2026-07-01. Cheapest canonical check found: dashboard.safe.ai/api/rli returns the CAIS
    # rows as plain JSON (what remotelabor.ai itself fetches) -- no bundle parsing needed.
    # Two standing cross-surface asymmetries, both already reconciled above: Gemini 3.7 Flash
    # is on Scale but still not in the CAIS API; Grok 4 is in CAIS but not on Scale. Scale
    # stores 2.502/2.501 for gpt-5.2 (medium)/Manus 1.5 -- rank tie-break jitter behind the
    # rendered 2.50, not a score change.
]


@st.cache_data
def load_rli_data():
    models = []
    for r in _RLI_RAW:
        models.append({
            'name': r['name'],
            'date': datetime.strptime(r['date'], '%Y-%m-%d'),
            'rli_score': r['rli_score'],
        })
    models.sort(key=lambda m: m['date'])

    # Frontier detection: running max
    max_score = -float('inf')
    for m in models:
        if m['rli_score'] > max_score:
            max_score = m['rli_score']
            m['is_frontier'] = True
        else:
            m['is_frontier'] = False

    return models


# ── CoBench (Anthropic internal AI R&D eval) ─────────────────────────────
# CoBench places a model at a historical snapshot of Anthropic's codebase,
# logs, internal messaging and docs and asks it to diagnose the root cause of
# an issue Anthropic engineers actually solved; 449 problems, model-graded
# against the root cause found in practice. The set is filtered for difficulty
# (mostly problems Mythos Preview failed at least once in three tries) and run
# at a 300k-token budget, so scores are not comparable to public AI R&D
# suites. Anthropic states that a model able to fully substitute for its
# research staff would score at least _RSI_SUBSTITUTION_BAR.
#
# Source: Anthropic, Redacted Risk Report (August 2026), §3.4.3 / Fig 3.4.3.A.
# Scores are read off that figure; Anthropic prints no table.
#
# `date_known` False marks a release date the record does not pin down, and is
# what puts the "~" in front of a date wherever it renders. Mythos Preview has
# no published release record (its date is carried over from AISI's narrow
# cyber figure, same as in aisi_cyber_tlo.csv); Model 2 (internal) is unreleased and its
# name redacted in the report, so its date is an estimate.

_RSI_SOURCE_URL = ("https://www-cdn.anthropic.com/f61d49fa5596956a5dec75fea0e973bf6a6a8378/"
                   "Redacted%20Risk%20Report%20August%202026%20.pdf#page=98.55")

# Anthropic's own bar: the score it thinks a model that could fully substitute
# for its Research Scientists and Engineers would reach.
_RSI_SUBSTITUTION_BAR = 85.0

_RSI_RAW = [
    {"name": "Claude Opus 4.6",       "date": "2026-02-05", "cobench": 15.6, "date_known": True},
    {"name": "Claude Mythos Preview", "date": "2026-04-07", "cobench": 54.8, "date_known": False},
    {"name": "Claude Mythos 5",       "date": "2026-06-09", "cobench": 50.3, "date_known": True},
    {"name": "Model 2 (internal)",    "date": "2026-07-06", "cobench": 62.8, "date_known": False},
]


def _rsi_date_label(m, fmt='%b %d, %Y'):
    """Date string, prefixed "~" when the release date isn't on the record."""
    return ("" if m['date_known'] else "~") + m['date'].strftime(fmt)


# The internal researcher survey — the other series in the report's section on
# substituting models for Anthropic staff (§3.4.2). Self-reported output
# multiple against no AI assistance, per frontier model. The rounds do not all
# report the same statistic on the same sample, so each point carries its own
# `note` on hover; the numbers are self-reported to one significant figure,
# hence the "~" everywhere they render. Opus 4 (May 2025) is deliberately not
# carried: that round reported no number, only that the result fell under the
# pre-set 3x median rule-out threshold, and a bound is not a point on a trend.

_RSI_SURVEY = [
    {"name": "Opus 4.5", "date": "2025-11-24", "uplift": 2.0,
     "note": "n=18, top ~30 staff by internal Claude Code use · median (mean 3.2x)"
             "<br>0 of 18 thought it crossed AI R&D-4"},
    {"name": "Opus 4.6", "date": "2026-02-05", "uplift": 2.0, "lo": 1.3, "hi": 8.0,
     "note": "n=16, broadened past superusers · median (mean 2.5x), range 1.3\u20138x"
             "<br>2 of 16 said a drop-in L4 was already possible; both revised on "
             "follow-up"},
    {"name": "Mythos Preview", "date": "2026-04-07", "uplift": 4.0,
     "note": "n=130, opt-in Slack poll · geometric mean, past week's work output"
             "<br>1 of 18 in a separate poll said a drop-in L4 already exists"},
    # No round was run for Model 2 (internal); ~4x is carried over from Mythos
    # Preview as the best available read. `estimated` draws it hollow as an
    # assumed value, but it is *in* the fit: with only three surveyed rounds,
    # ignoring the one flat reading available overstates the slope.
    {"name": "Model 2 (internal)", "date": "2026-07-06", "uplift": 4.0,
     "estimated": True,
     "note": "no survey run · ~4x carried over from Mythos Preview"},
]

# The survey fan's position CI: the fitted multiple divided and multiplied by
# this. Self-reports to one significant figure don't support anything tighter.
_RSI_SURVEY_POS_FACTOR = 1.5

# The survey fan runs a year past the last round rather than to the tab's
# *Project through* year. Three coarse points 4.5 months apart compound to
# ~10^6x by end-2028, which squashes the data itself into the bottom decile of
# a log axis; the CoBench half keeps the sidebar horizon.
_RSI_SURVEY_HORIZON_DAYS = 365


@st.cache_data
def load_rsi_survey():
    rows = [{**r, 'date': datetime.strptime(r['date'], '%Y-%m-%d')}
            for r in _RSI_SURVEY]
    rows.sort(key=lambda r: r['date'])
    return rows


# ── Research direction (Anthropic's Claude Code detour study) ────────────
# The third substitution series: Anthropic took `_RSI_DIR_N` moments from real
# January–March 2026 Claude Code research sessions where a researcher took a
# detour that sent the session off-track, showed each model only the work
# preceding the detour, and asked what it would do next. A separate Claude
# with the whole session in view — including how it was eventually resolved —
# judged whether the model's next step or the human's was better.
#
# Not apples-to-apples by construction: the turns were *selected* for having
# room for improvement, so the rate is not "how often Claude out-researches a
# human", it is how often it out-picks one at a turn already known to be a
# wrong one.
#
# `tie` is the share the judge called neither better; it is carried for the
# hover only. The series projected is `better`, the figure's own headline.

_RSI_DIR_SOURCE_URL = "https://www.anthropic.com/institute/recursive-self-improvement"
_RSI_DIR_N = 129                  # sampled detour turns

# The bar, and the only one the source names: the figure's annotated
# practical ceiling, which is what an *oracle* scores — a model shown the
# complete session, resolution included, judged against the same researchers.
# Reaching it means picking the better next step as often as something that
# already knows how the session ended.
_RSI_DIR_TARGET = 90.0

# 80% CI on where the frontier stands today, in points. The study reports
# whole percentages off n=129, whose binomial standard error at ~60% is ~4.3
# points, so ±5 is the sampling noise on a single figure rather than a guess.
_RSI_DIR_POS_CI = 5.0

_RSI_DIR_RAW = [
    {"name": "Claude Haiku 3",        "date": "2024-03-07", "better": 22.0, "tie": 10.0},
    {"name": "Claude Sonnet 4",       "date": "2025-05-22", "better": 48.0, "tie": 11.0},
    {"name": "Claude Sonnet 4.5",     "date": "2025-09-29", "better": 50.0, "tie": 11.0},
    {"name": "Claude Haiku 4.5",      "date": "2025-10-15", "better": 45.0, "tie": 11.0},
    {"name": "Claude Opus 4.5",       "date": "2025-11-24", "better": 51.0, "tie": 10.0},
    {"name": "Claude Opus 4.6",       "date": "2026-02-05", "better": 55.0, "tie": 14.0},
    {"name": "Claude Sonnet 4.6",     "date": "2026-02-17", "better": 45.0, "tie": 13.0},
    # Same unpublished date the CoBench series carries for this model, hence
    # `date_known` False. The figure prints Opus 4.7 above it, but that figure
    # is not date-ordered either (it puts Sonnet 4.6 above Opus 4.6, which
    # shipped twelve days earlier), so the dates stand and Opus 4.7 lands off
    # the running max.
    {"name": "Claude Mythos Preview", "date": "2026-04-07", "better": 64.0, "tie": 9.0,
     "date_known": False},
    {"name": "Claude Opus 4.7",       "date": "2026-04-16", "better": 59.0, "tie": 12.0},
]


@st.cache_data
def load_rsi_direction():
    rows = [{**r, 'date': datetime.strptime(r['date'], '%Y-%m-%d'),
             'date_known': r.get('date_known', True)} for r in _RSI_DIR_RAW]
    rows.sort(key=lambda r: r['date'])

    max_score = -float('inf')
    for r in rows:
        r['is_frontier'] = r['better'] > max_score
        if r['is_frontier']:
            max_score = r['better']
    return rows


# ── Code merged per person (Anthropic's own repositories) ────────────────
# The fourth substitution series, and the only one that counts output rather
# than scoring a benchmark: lines of code merged per active contributor each
# quarter, as a multiple of the pre-2025 average. Per-PR line counts are
# capped at the 99th percentile and an "active contributor" is a distinct
# author in the trailing twelve months, both per the figure's own footnote.
#
# Digitized from the figure in *When AI builds itself*, which prints its own
# bar labels from 2025Q1 on; the earlier bars carry no label and are read off
# the axis. Those bars are the normalization's own baseline — they average to
# 1 by construction — so they are charted but not fitted
# (`_RSI_CODE_FIT_FROM`), and their mean is the calibration guard on the
# digitization. The post states the 2026Q2 figure in prose, which pins the
# other end.
#
# `partial` marks the figure's hatched final bar: it averages only the days
# observed when the post went up. It is fitted, at its quarter's midpoint
# like every other bar, which dates it later than the days it covers and so
# reads the ramp slow rather than hot.

_RSI_CODE_SOURCE_URL = _RSI_DIR_SOURCE_URL

# The bar: one contributor merging what that many did before 2025.
_RSI_CODE_TARGET = 30.0

_RSI_CODE_FIT_FROM = datetime(2025, 1, 1)

# Position CI: the fit's own residual scatter (80%), rather than the survey's
# one-significant-figure allowance — this series is counted, not self-reported.
_RSI_CODE_POS_FACTOR = 1.3

# Where the x-axis opens. The bars before it are flat by construction and
# only stretch the axis; they stay in the data, where their mean is the
# calibration guard on the digitization.
_RSI_CODE_CHART_FROM = datetime(2024, 1, 1)

# Same reason the survey fan stops short of the tab's horizon: a multiple on a
# ~160-day doubling time leaves the chart long before the projection year, and
# squashes the measured bars into the bottom of a log axis.
_RSI_CODE_HORIZON_DAYS = 365

_RSI_CODE_RAW = [
    ("2021Q2", 1.40), ("2021Q3", 1.15), ("2021Q4", 0.87),
    ("2022Q1", 1.09), ("2022Q2", 1.17), ("2022Q3", 0.69), ("2022Q4", 0.70),
    ("2023Q1", 1.04), ("2023Q2", 1.07), ("2023Q3", 0.98), ("2023Q4", 0.85),
    ("2024Q1", 1.00), ("2024Q2", 0.85), ("2024Q3", 0.87), ("2024Q4", 0.89),
    ("2025Q1", 1.20), ("2025Q2", 1.50), ("2025Q3", 1.90), ("2025Q4", 2.50),
    ("2026Q1", 5.80), ("2026Q2", 8.00),
]
_RSI_CODE_PARTIAL = "2026Q2"      # the hatched bar: a quarter still running


def _rsi_code_mid(q):
    """Midpoint datetime of a '2026Q2' quarter label."""
    y, k = int(q[:4]), int(q[-1])
    start = datetime(y, 3 * k - 2, 1)
    end = datetime(y + (k == 4), 3 * k % 12 + 1, 1)
    return start + (end - start) / 2


def _rsi_code_hover(r):
    """One point's hover: the quarter, its multiple and why it draws as it does."""
    if r['partial']:
        tail = "<br>partial quarter — the days observed so far"
    elif not r['fitted']:
        tail = "<br>pre-2025 baseline — charted, not fitted"
    else:
        tail = ""
    return f"{r['q'][:4]} {r['q'][4:]}<br>{r['mult']:g}x pre-2025 average" + tail


@st.cache_data
def load_rsi_code():
    rows = [{'q': q, 'date': _rsi_code_mid(q), 'mult': v,
             'partial': q == _RSI_CODE_PARTIAL,
             'fitted': _rsi_code_mid(q) >= _RSI_CODE_FIT_FROM}
            for q, v in _RSI_CODE_RAW]
    rows.sort(key=lambda r: r['date'])
    return rows


@st.cache_data
def load_rsi_data():
    models = [{
        'name': r['name'],
        'date': datetime.strptime(r['date'], '%Y-%m-%d'),
        'cobench': r['cobench'],
        'date_known': r['date_known'],
    } for r in _RSI_RAW]
    models.sort(key=lambda m: m['date'])

    # Frontier detection: running max, same convention as every other tab.
    max_score = -float('inf')
    for m in models:
        m['is_frontier'] = m['cobench'] > max_score
        if m['is_frontier']:
            max_score = m['cobench']

    return models


# ── AISI narrow cyber tasks ──────────────────────────────────────────────
# Average success rate on 70 of AISI's narrow cyber tasks. Unlike every other
# feed in this app, AISI publishes no numbers for this chart -- the values in
# aisi_cyber_narrow.csv were digitized from the published figure by pixel
# analysis. See the CSV header for calibration and validation details, and
# _UKC_PROVENANCE below for the caveat surfaced in the UI.

_UKC_PROVENANCE = (
    "Values digitized from AISI's published figure by pixel analysis — AISI releases "
    "no underlying numbers for this chart. Dates are inferred from marker positions, not "
    "published release dates. Calibration reproduces AISI's own printed lag annotations "
    "(4.3mo, 5.0mo) to within 0.1 month, and derived dates match known releases to within "
    "~1 day, but treat all values as approximate."
)

# The figure contains only closed-weight US models and open-weight Chinese
# models -- no US open-weight, no Chinese closed. Country and openness are
# therefore perfectly confounded, so the frontier is defined by weights
# (the published framing) rather than by country. The two Chinese models here
# are also the only open-weight ones, so "China" and "open-weight" name the same
# two points -- worth stating wherever the tab says "China". Folded into the
# fine-print caption.
_UKC_CONFOUND_PLAIN = (
    "\"China\" here means the two Chinese open-weight models in AISI's figure; it contains "
    "no US open-weight and no Chinese closed-weight models, so country and openness cannot "
    "be separated in this data."
)


def _ukc_mtime():
    p = os.path.join(os.path.dirname(__file__), 'aisi_cyber_narrow.csv')
    return os.path.getmtime(p)


@st.cache_data
def load_ukcyber(_mtime=None):
    """Load digitized AISI narrow-cyber-task success rates.

    Returns models sorted by date, each flagged `is_frontier` if it set a new
    running-max success rate among closed-weight models.
    """
    csv_path = os.path.join(os.path.dirname(__file__), 'aisi_cyber_narrow.csv')
    with open(csv_path, 'r') as f:
        lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    reader = csv.DictReader(lines)

    models = []
    for r in reader:
        score_str = (r.get('success_rate') or '').strip()
        date_str = (r.get('date') or '').strip()
        if not score_str or not date_str:
            continue
        try:
            score = float(score_str)
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        models.append({
            'name': (r.get('model') or '').strip(),
            'date': date,
            'cyber_score': score,
            'organization': (r.get('organization') or '').strip(),
            'country': (r.get('country') or '').strip(),
            'weights': (r.get('weights') or '').strip().lower(),
        })
    models.sort(key=lambda m: m['date'])

    # Frontier = running max among closed-weight models. Open-weight models are
    # the subject being measured against it, so they never define it.
    max_score = -float('inf')
    for m in models:
        if m['weights'] == 'closed' and m['cyber_score'] > max_score:
            max_score = m['cyber_score']
            m['is_frontier'] = True
        else:
            m['is_frontier'] = False

    return models


_UKC_TLO_STEPS = 32          # "The Last Ones" is a 32-step attack chain
_UKC_TLO_FILE = 'aisi_cyber_tlo.csv'


def _ukc_tlo_mtime():
    p = os.path.join(os.path.dirname(__file__), _UKC_TLO_FILE)
    return os.path.getmtime(p)


@st.cache_data
def load_ukcyber_tlo(_mtime=None):
    """Load AISI/CAISI cyber-range ("The Last Ones") average steps completed.

    The long-horizon counterpart to `load_ukcyber()`: same institution and eval
    protocol, but it scores autonomous end-to-end attack execution instead of
    isolated skills, and it produces the wide end of AISI's "4 to 7 months"
    open-weight lag where the narrow tasks produce the narrow end.

    Deliberately returns the same shape as `load_ukcyber()`, with `cyber_score`
    carrying steps (0-32) rather than percent, so `_ukc_frontier_crossing()` and
    `ukc_lag_rows()` work on it unchanged. `steps` is kept as an alias for
    callers that want the unit to be explicit at the call site.
    """
    csv_path = os.path.join(os.path.dirname(__file__), _UKC_TLO_FILE)
    with open(csv_path, 'r') as f:
        lines = [ln for ln in f if not ln.lstrip().startswith('#')]
    reader = csv.DictReader(lines)

    models = []
    for r in reader:
        steps_str = (r.get('steps') or '').strip()
        date_str = (r.get('date') or '').strip()
        if not steps_str or not date_str:
            continue
        try:
            steps = float(steps_str)
            date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            continue
        models.append({
            'name': (r.get('model') or '').strip(),
            'date': date,
            'cyber_score': steps,
            'steps': steps,
            'organization': (r.get('organization') or '').strip(),
            'country': (r.get('country') or '').strip(),
            'weights': (r.get('weights') or '').strip().lower(),
        })
    models.sort(key=lambda m: m['date'])

    # Same frontier rule as the narrow tasks: running max over closed-weight
    # models only, since the open ones are the subject being measured.
    max_score = -float('inf')
    for m in models:
        if m['weights'] == 'closed' and m['cyber_score'] > max_score:
            max_score = m['cyber_score']
            m['is_frontier'] = True
        else:
            m['is_frontier'] = False

    return models


def ukc_tlo_lag_rows():
    """TLO models plus their lag rows, computed exactly as the narrow ones are.

    Shared by the headline callout and the cross-check section so the two can't
    drift apart on which frontier or lag convention they use.
    """
    tlo_all = load_ukcyber_tlo(_ukc_tlo_mtime())
    return tlo_all, ukc_lag_rows(tlo_all, [m for m in tlo_all if m['is_frontier']])


def ukc_open_only_on_tlo(narrow_lag_rows, tlo_lag_rows):
    """The newest open-weight model the cyber range has measured but the narrow
    tasks have not -- or None when the two suites cover the same models.

    Kimi K3 is the case this exists for. AISI/CAISI ran only a selective set on
    it (ExploitBench + the TLO range), not the 70-task narrow suite, so it has
    no point on the main chart and would otherwise not surface until the
    cross-check section at the very bottom of the tab. Returns None once the
    narrow suite catches up, so the callout is a data-coverage notice that
    disappears on its own rather than a permanent panel.

    Only reports a model newer than every open-weight model on the narrow
    chart: an *older* gap in TLO coverage is a curiosity, not a case of the
    headline chart being out of date.
    """
    narrow_names = {r['name'] for r in narrow_lag_rows}
    missing = [r for r in tlo_lag_rows if r['name'] not in narrow_names]
    if not missing:
        return None
    newest = max(missing, key=lambda r: r['date'])
    if narrow_lag_rows and newest['date'] <= max(r['date'] for r in narrow_lag_rows):
        return None
    return newest


def _ukc_frontier_match_for_score(frontier, score):
    """First closed-frontier model that matched or beat `score`.

    This is the *optimistic* end of the lag bracket -- see
    `_ukc_frontier_crossing`. AISI's published annotations use this convention,
    so it is also the calibration check on the digitization.

    Returns None if no frontier model reaches it (the model is ahead, not behind).
    """
    for m in frontier:
        if m['cyber_score'] >= score:
            return m
    return None


def _ukc_frontier_below_for_score(frontier, score):
    """Last closed-frontier model still *below* `score` (pessimistic bracket end)."""
    below = [m for m in frontier if m['cyber_score'] < score]
    return below[-1] if below else None


def _ukc_frontier_crossing(frontier, score):
    """Date the closed frontier reached `score`, interpolated between the two
    bracketing released models.

    Snapping to the next model up would equate scores that are far apart:
    DeepSeek-V4-Pro's 55.7% sits in a 10-point gap between GPT-5 (52.5%,
    Aug 2025) and Opus 4.5 (62.6%, Nov 2025). Calling it "as good as Opus 4.5"
    credits it with ~7 points it does not have and understates its lag by
    ~2.4 months. Interpolating places the crossing where the frontier plausibly
    passed that level instead.

    The honest caveat is that no model was released in that gap, so the crossing
    date is an estimate; `ukc_lag_rows` carries the bracketing models alongside
    it so the width of that uncertainty stays visible.

    Returns (crossing_date, below_model, above_model); crossing_date is None if
    the frontier never reaches the score.
    """
    above = _ukc_frontier_match_for_score(frontier, score)
    if above is None:
        return None, None, None
    below = _ukc_frontier_below_for_score(frontier, score)
    if below is None:
        # Score is at or under the first frontier point -- nothing to interpolate.
        return above['date'], None, above
    span = above['cyber_score'] - below['cyber_score']
    if span <= 0:
        return above['date'], below, above
    frac = (score - below['cyber_score']) / span
    crossing = below['date'] + timedelta(
        days=(above['date'] - below['date']).days * frac)
    return crossing, below, above


_UKC_DAYS_PER_MONTH = 30.44


def ukc_lag_rows(models, frontier):
    """Lag of each open-weight model behind the closed frontier.

    `lag_months` is the point estimate, from the interpolated crossing.
    `lag_lo`/`lag_hi` bracket it using the two models the score falls between:
    the lower bound credits the open model with matching the stronger model
    above it, the upper bound only credits it with beating the weaker model
    below. No model was released between them, so that width is real
    uncertainty rather than noise.
    """
    rows = []
    for m in models:
        if m['weights'] != 'open':
            continue
        crossing, below, above = _ukc_frontier_crossing(frontier, m['cyber_score'])
        if crossing is None:
            rows.append({**m, 'match_date': None, 'below_name': None,
                         'above_name': None, 'lag_days': None, 'lag_months': None,
                         'lag_lo': None, 'lag_hi': None})
            continue
        lag_days = (m['date'] - crossing).days
        rows.append({
            **m,
            'match_date': crossing,
            'below_name': below['name'] if below else None,
            'above_name': above['name'] if above else None,
            'lag_days': lag_days,
            'lag_months': lag_days / _UKC_DAYS_PER_MONTH,
            'lag_lo': (m['date'] - above['date']).days / _UKC_DAYS_PER_MONTH if above else None,
            'lag_hi': (m['date'] - below['date']).days / _UKC_DAYS_PER_MONTH if below else None,
        })
    return rows


_UKC_TARGET = 90.0


def ukc_target_eta(models, frontier, target=_UKC_TARGET):
    """When do open-weight models reach `target`% success?

    Modelled as the closed frontier's crossing date plus the measured
    open-weight lag -- the same frontier+lag structure the tab is built on,
    and the only approach the data supports. Fitting a trend through the
    open-weight points alone would extrapolate from two models released 53
    days apart; see `ukc_target_eta_direct` for that as a cross-check.

    Returns None if the frontier never reaches the target.
    """
    crossing, below, above = _ukc_frontier_crossing(frontier, target)
    if crossing is None:
        return None
    lags = [r['lag_months'] for r in ukc_lag_rows(models, frontier)
            if r['lag_months'] is not None]
    if not lags:
        return None
    lag_lo, lag_hi = min(lags), max(lags)
    return {
        'target': target,
        'frontier_date': crossing,
        'frontier_between': (below['name'] if below else None,
                             above['name'] if above else None),
        'lag_lo': lag_lo,
        'lag_hi': lag_hi,
        'date_lo': crossing + timedelta(days=lag_lo * _UKC_DAYS_PER_MONTH),
        'date_hi': crossing + timedelta(days=lag_hi * _UKC_DAYS_PER_MONTH),
    }


def ukc_target_eta_direct(models, target=_UKC_TARGET):
    """Cross-check: extrapolate the open-weight points themselves, in logit space.

    Deliberately not the headline -- with only two open-weight models this
    slope is very sensitive to either point. Useful only to check the
    lag-based estimate lands in the same season.
    """
    open_models = sorted([m for m in models if m['weights'] == 'open'],
                         key=lambda m: m['date'])
    if len(open_models) < 2:
        return None
    base = open_models[0]['date']
    days = np.array([(m['date'] - base).days for m in open_models], dtype=float)
    lg = _logit(np.array([m['cyber_score'] / 100 for m in open_models]))
    params = fit_line(days, lg)
    if params[1] <= 0:
        return None
    needed = (_logit(target / 100) - params[0]) / params[1]
    return base + timedelta(days=float(needed))


# ── Data center data (Epoch AI Frontier Data Centers) ────────────────────

import bisect

# 2-month training run (2 × 30-day months), used to turn a site's 8-bit OP/s
# throughput into total operations over a two-month run. The 6-month variant
# backs the longer-run capacity metric; it is the same arithmetic with a 3×
# longer window, so its FLOP numbers are exactly 3× the 2-month ones.
_DAYS_2MO = 2 * 30
_SECONDS_2MO = _DAYS_2MO * 24 * 3600
_DAYS_6MO = 6 * 30
_SECONDS_6MO = _DAYS_6MO * 24 * 3600
# A model ships ~1mo after its training run finishes (post-training, evals,
# safety), calibrated to observed train-finish → announce gaps. Used both to
# date model runs and to expand a buildout milestone into its DC-online /
# training / model-out timeline.
_CC_RUN_COMPLETION_LAG = timedelta(days=30)


def _dc_milestone_dates(record_date, shift_days=0, run_days=None):
    """Tooltip date lines for a buildout milestone. The plotted date is the site's
    DC-available date shifted forward by the chosen timing (`shift_days`), so
    reconstruct the base and expand into the milestones the timing dropdown
    offers: DC online, and — only when `run_days` names a training-run length,
    i.e. the capacity metric is a train-OP one — training done / model out
    for a run of that length. Other metrics assume no run, so they get the
    DC-online line alone.
    """
    base = record_date - timedelta(days=shift_days)
    out = f"DC online ~{base:%b %d, %Y}"
    if run_days is not None:
        done = base + timedelta(days=run_days)
        out += (f"<br>Training done ({run_days // 30}mo run) ~{done:%b %d, %Y}"
                f"<br>Model out ~{(done + _CC_RUN_COMPLETION_LAG):%b %d, %Y}")
    return out


# Fraction of peak throughput actually realized over a real training run.
_DC_UTILIZATION = 0.3

def _dc_mtime():
    p = os.path.join(os.path.dirname(__file__), 'data_center_timelines.csv')
    return os.path.getmtime(p)


def _dc_clean_owner(s):
    """Strip Epoch '#confident'/'#speculative' confidence tags from an owner field."""
    return (s or '').split('#')[0].strip()


# Distinct Epoch labels that are one company for presentation. Applied to the
# derived company label, never to the CSVs.
#
# Google is the load-bearing case: every Google site is Owner="Google", but only
# some also carry Users="Google DeepMind #speculative" and the rest leave Users
# blank, so company_for() (user-first, owner-fallback) split one fleet in two on
# but whether Epoch filled an optional, self-tagged-speculative cell — Lancaster,
# with Users blank, lists the same TPU v5e/v5p/v6e/v7 as the tagged sites. The
# split put Google blue on the smaller of two lines, made the pooled Columbus
# cluster depend on both its Google sites happening to share a tag, and had the
# quarterly table reporting the minority series (2.6x low by 2027Q4). The rest of
# the app already merges them: _cc_lab_for_site() maps owner "Google*" to Google,
# and the ECI tabs substring-match "Google" for the same reason.
_DC_COMPANY_ALIASES = {
    "Google DeepMind": "Google",
}


@st.cache_data
def load_data_centers(_mtime=None):
    """Load Epoch's per-data-center capacity timelines plus an owner→company map.

    Returns a list of dicts: {name, company, attributed, country, points:[{date,
    status, h100, it_power, power, perf, cost}, ...]} with points sorted by date. Metric
    values are floats or None when missing. `attributed` is False when the
    company label came from the site-name fallback rather than a recorded user
    or owner.
    """
    base = os.path.dirname(__file__)

    # Operator (primary user) and owner maps from the data_centers metadata file.
    # We attribute each site to the AI lab operating it (its primary listed user),
    # falling back to the facility owner, then to the site-name token.
    user_by_dc = {}
    users_by_dc = {}
    owner_by_dc = {}
    country_by_dc = {}
    dc_meta_path = os.path.join(base, 'data_centers.csv')
    if os.path.exists(dc_meta_path):
        with open(dc_meta_path, 'r') as f:
            for r in csv.DictReader(f):
                name = (r.get('Name') or '').strip()
                if not name:
                    continue
                owner_by_dc[name] = _dc_clean_owner(r.get('Owner', ''))
                # Users may be a comma-separated list; keep them all (for
                # shared-tenancy attribution) and the primary (first) for the
                # site's single label.
                users = [u for u in (_dc_clean_owner(x) for x in
                                     (r.get('Users', '') or '').split(','))
                         if u]
                users_by_dc[name] = users
                user_by_dc[name] = users[0] if users else ''
                country_by_dc[name] = (r.get('Country') or '').strip()

    def company_for(dc_name):
        label = (user_by_dc.get(dc_name, '')
                 or owner_by_dc.get(dc_name, '')
                 # Fallback for colocation sites with no listed user/owner:
                 # use the first token of the site name (QTS, DayOne, EdgeCore…).
                 or (dc_name.split()[0] if dc_name.split() else dc_name))
        return _DC_COMPANY_ALIASES.get(label, label)

    def attributed_for(dc_name):
        """True when Epoch actually records who uses or owns the site.

        False means company_for() fell through to the site-name token, so the
        label is a landlord read off the building's name and the tenant actually
        training on the hardware is unknown. The charts mark those companies —
        see _dc_unattributed_companies().
        """
        return bool(user_by_dc.get(dc_name, '') or owner_by_dc.get(dc_name, ''))

    def _num(r, key):
        v = (r.get(key, '') or '').strip().replace(',', '')
        if not v:
            return None
        try:
            return float(v)
        except (ValueError, TypeError):
            return None

    series = {}
    with open(os.path.join(base, 'data_center_timelines.csv'), 'r') as f:
        for r in csv.DictReader(f):
            dname = (r.get('Data center') or '').strip()
            ds = (r.get('Date') or '').strip()
            if not dname or not ds:
                continue
            try:
                d = datetime.strptime(ds, '%Y-%m-%d')
            except (ValueError, TypeError):
                continue
            perf = _num(r, 'Performance (8-bit OP/s)')
            # Total 8-bit OPs from a 2-month run at this throughput, derated by
            # realized utilization.
            train_flop = (perf * _SECONDS_2MO * _DC_UTILIZATION
                          if perf is not None else None)
            train_flop_6mo = (perf * _SECONDS_6MO * _DC_UTILIZATION
                              if perf is not None else None)
            series.setdefault(dname, []).append({
                'date': d,
                'status': r.get('Construction status', ''),
                'h100': _num(r, 'H100 equivalents'),
                'it_power': _num(r, 'IT power (MW)'),
                'power': _num(r, 'Power (MW)'),
                'perf': perf,
                'train_flop': train_flop,
                'train_flop_6mo': train_flop_6mo,
                # How many GPT-5-scale (2e25 FLOP) / Mythos-scale (1e27 FLOP)
                # training runs the site's 2-month capacity could produce.
                # Displayed as *time to train one* (kind 'traintime',
                # days = _DAYS_2MO / runs), but stored as runs-per-2mo so that
                # every "largest data center" aggregation in the tab — envelope,
                # per-company max, ranking — stays a plain max. Bigger number =
                # faster site, so the ordering is identical either way.
                'gpt5s': train_flop / 2e25 if train_flop is not None else None,
                'mythos': train_flop / 1e27 if train_flop is not None else None,
                'cost': _num(r, 'Total capital cost (2025 USD billions)'),
            })

    dcs = []
    for name, pts in series.items():
        pts.sort(key=lambda p: p['date'])
        token = name.split()[0] if name.split() else name
        # Raw attribution, for tabs that show who owns the building separately
        # from who trains in it. Both fall through to the site-name token like
        # company_for(); 'tenant' is exactly company_for()'s label.
        operator = owner_by_dc.get(name, '') or token
        # Every listed user, aliased and deduped in Epoch's order — the
        # shared-tenancy attribution ('users' empty when Epoch names none).
        users = []
        for u in users_by_dc.get(name, ()):
            u = _DC_COMPANY_ALIASES.get(u, u)
            if u not in users:
                users.append(u)
        dcs.append({'name': name, 'company': company_for(name),
                    'attributed': attributed_for(name), 'points': pts,
                    'operator': _DC_COMPANY_ALIASES.get(operator, operator),
                    'tenant': company_for(name), 'users': users,
                    'country': country_by_dc.get(name, '')})
    dcs.sort(key=lambda dc: dc['name'])
    return dcs


# Metric options for the data-center tab: label → (point key, log-scale default, formatter kind)
_DC_METRICS = {
    # Every metric is stored, plotted, hovered and tabulated in raw units; the
    # axis ticks are labelled by `_dc_axis_ticks` the same way `_dc_fmt_value`
    # labels a value, so no chart can read in different units than its labels.
    "Compute (H100-equiv)": {"key": "h100", "log": True, "kind": "h100"},
    "Power (MW)": {"key": "power", "log": False, "kind": "mw"},
    "IT power (MW)": {"key": "it_power", "log": False, "kind": "mw"},
    "Capital cost ($B)": {"key": "cost", "log": False, "kind": "cost"},
    "Performance (8-bit OP/s)": {"key": "perf", "log": True, "kind": "sci"},
    # `run_days` is the training-run window a metric assumes; it sizes the
    # timing shifts below. Metrics that don't depend on run length default to
    # the 2-month convention used everywhere else in the tab.
    "2mo train log OP": {"key": "train_flop", "log": True, "kind": "flop",
                       "run_days": _DAYS_2MO},
    "6mo train log OP": {"key": "train_flop_6mo", "log": True, "kind": "flop",
                       "run_days": _DAYS_6MO},
    "Capacity (time to GPT-5)": {"key": "gpt5s", "log": True, "kind": "traintime"},
    "Capacity (time to Mythos)": {"key": "mythos", "log": True, "kind": "traintime"},
}

# Timing options: label → days the DC-available date is shifted forward to date
# the chosen milestone.
#   • DC construction   — no shift (the site's availability date)
#   • Training finished  — +one training run (the metric's `run_days`: a run
#                          started at availability finishes that much later)
#   • Model release      — + run + ~1mo post-training/eval lag (matches the
#                          Compute/capabilities/diffusion tab's model-release dating)
_DC_TIMING_OPTIONS = (
    "Data center construction",
    "Training run finished",
    "Model release",
)


def _dc_timing_shift(label, run_days=_DAYS_2MO):
    """Days to shift a site's availability date to reach the chosen milestone.

    `run_days` comes from the selected metric, so the 6-month FLOP metric dates
    its models six months out rather than two.
    """
    if label == "Training run finished":
        return run_days
    if label == "Model release":
        return run_days + _CC_RUN_COMPLETION_LAG.days
    return 0


# Stable colors for the most common companies; others fall back to a palette.
_DC_COLORS = {
    "Google": "#4285F4", "Meta": "#0866FF", "Microsoft": "#7CBB00",
    "OpenAI": "#10A37F", "Oracle": "#C74634", "Amazon": "#FF9900",
    "CoreWeave": "#FF4D4D", "SpaceXAI": "#1DA1F2", "Anthropic": "#D97757",
    "Softbank": "#7A2E8E", "Alibaba": "#FF6A00", "Fluidstack": "#00B5AD",
    "Nscale": "#5C6BC0", "G42": "#16A085",
}
_DC_PALETTE = ["#888888", "#E377C2", "#8C564B", "#BCBD22", "#17BECF",
               "#9467BD", "#2CA02C", "#D62728", "#1F77B4", "#FF7F0E"]

# Companies excluded from all data-center views (colocation/neutral-host
# providers plus others not wanted on the chart).
# Sites that could plausibly run ONE training job together. The cross-site link
# carries the data-parallel gradient all-reduce, so pooling needs either metro
# fibre or a purpose-built long-haul fabric; merely sharing an owner does not
# qualify. Entries are (cluster label, basis, site names), basis being:
#   'proximity' — same campus or metro, read off the Address column
#   'fabric'    — far apart, but joined by an announced training fabric
#   'plausible' — one region, every pair of sites in it within about the span
#                 of that announced fabric, but no link announced: what a
#                 company could wire together if it chose to
# The bases are nested, weakest first, and the tab picks how far down the list
# to go — see _dc_network_site_clusters(). A 'plausible' region therefore has to
# *contain* any proximity or fabric cluster it touches, or turning it on would
# split an already-pooled group instead of widening it; TestDcNetworkClusters
# checks that. Anything absent is its own cluster and never pools. Addresses
# are too irregular to cluster automatically (27 of 78 don't parse, and the Cedar Rapids
# pair sits in two differently-named municipalities), so this is curated by
# hand; TestDcNetworkClusters checks every name against the live CSV so a data
# refresh that renames a site fails loudly instead of silently un-clustering it.
#
# A cluster label names the geography or the fabric, never the tenant — the
# Fairwater pair pools under OpenAI, Epoch's first-listed user for both sites,
# not under Microsoft who owns them.
#
# Clusters are geography, not ownership, and pooling happens strictly within one
# company — so a cluster whose sites belong to different companies, or whose
# sites are hidden by _dc_hidden_companies(), is inert until that changes. Only
# Cedar Rapids is inert today, and for the first reason: its two sites are a
# Google one and a QTS one with no recorded user, so they never merge. Richmond
# (three QTS sites) and San Antonio (two Microsoft ones) used to be inert for
# the second reason and now pool, since those hosts cleared the size threshold
# and are charted. A cluster stays listed either way — the geography is real
# even where the attribution is not.
_DC_NETWORK_CLUSTERS = (
    ("Memphis, TN", 'proximity', ("Colossus 1", "Colossus 2")),
    ("Abilene, TX", 'proximity', ("OpenAI Stargate Abilene",
                                  "Crusoe Abilene Expansion",
                                  "OpenAI Stargate Shackelford")),
    ("Columbus, OH", 'proximity', ("Meta Prometheus", "Google New Albany",
                                   "Google Columbus")),
    ("Cedar Rapids, IA", 'proximity', ("Google Cedar Rapids",
                                       "QTS Cedar Rapids")),
    ("Richmond, VA", 'proximity', ("QTS Richmond 1", "QTS Richmond 2",
                                   "QTS Richmond 3")),
    ("San Antonio, TX", 'proximity', ("Microsoft SAT14", "Microsoft SAT40",
                                      "Vantage TX1")),
    ("Eagle Mountain, UT", 'proximity', ("Meta Eagle Mountain",
                                         "QTS Eagle Mountain")),
    ("Fairwater AI WAN", 'fabric', ("Microsoft Fairwater Wisconsin",
                                    "Microsoft Fairwater Atlanta")),
    # Regional groups: no link announced, but every pair inside a group sits
    # within roughly the ~1,200 km span of the AI WAN above — the one fabric
    # anyone has actually announced, and therefore the evidence that a link that
    # long is buildable. Refusing a shorter unannounced hop than one already
    # being built would be the inconsistent choice; crediting it as *usable*
    # today would be the wrong one, hence a separate, non-default level.
    #
    # Listed only where the group changes something: at least one company must
    # have two sites in it. An inert regional group pools nothing, unlike the
    # metro clusters above, which are kept for the geography even when inert.
    # Where a site is in range of two groups it goes to the one its own company
    # already occupies (Anthropic-Amazon New Carlisle joins the Mid-South rather
    # than the Great Lakes for that reason) — except when it sits in a
    # proximity or fabric cluster, which pins it and everything else in that
    # cluster to one group (Meta Prometheus stays in the Great Lakes with the
    # two Google sites it shares Columbus with). Left out for being at or past
    # the radius from every group that would use them: OpenAI Stargate New
    # Mexico, and the Phoenix sites.
    ("Great Lakes", 'plausible', ("Google Columbus", "Google New Albany",
                                  "Google Lancaster", "Google Fort Wayne",
                                  "Meta Prometheus", "OpenAI Stargate Michigan",
                                  "OpenAI Stargate Lordstown",
                                  "OpenAI Stargate Wisconsin")),
    ("Mid-South", 'plausible', ("Colossus 1", "Colossus 2", "Amazon Ridgeland",
                                "Amazon Madison Mega Site",
                                "Anthropic-Amazon New Carlisle",
                                "Meta Hyperion", "Meta Gallatin",
                                "Meta Huntsville", "Meta Montgomery",
                                "Meta Aiken", "Meta Jeffersonville")),
    ("Great Plains", 'plausible', ("Google Omaha", "Google Papillion",
                                   "Google Council Bluffs (East)",
                                   "Google Lincoln", "Google Kansas City East",
                                   "Google Cedar Rapids", "QTS Cedar Rapids",
                                   "Meta Sarpy", "Meta Rosemount",
                                   "Microsoft Project Osmium")),
    ("Mid-Atlantic", 'plausible', ("Google Bristow", "Google Arcola",
                                   "STACK Infrastructure NVA02",
                                   "QTS Richmond 1", "QTS Richmond 2",
                                   "QTS Richmond 3", "CoreWeave Chester VA",
                                   "AWS Berwick", "Anthropic Lake Mariner",
                                   "Microsoft-Nebius New Jersey")),
    ("Texas & Oklahoma", 'plausible', ("Google Midlothian", "Google Red Oak",
                                       "Google Pryor (North)", "Goodnight",
                                       "CoreWeave Denton TX",
                                       "CoreWeave Muskogee OK",
                                       "Coreweave Helios",
                                       "OpenAI Stargate Abilene",
                                       "OpenAI Stargate Shackelford",
                                       "Crusoe Abilene Expansion",
                                       "OpenAI Stargate Milam", "Meta Temple",
                                       "Microsoft SAT14", "Microsoft SAT40",
                                       "Vantage TX1")),
    ("Mountain West", 'plausible', ("Meta Eagle Mountain", "QTS Eagle Mountain",
                                    "Meta Kuna", "Meta Cheyenne")),
    ("Pacific Northwest", 'plausible', ("Google The Dalles",
                                        "Google Storey County",
                                        "Meta-QTS Hillsboro 2")),
)

# The cluster bases, weakest first; each level admits every basis before it.
_DC_NETWORK_LEVELS = ('proximity', 'fabric', 'plausible')


# ── Buildout by country ──────────────────────────────────────────────────
# Country is a property of the building, read off Epoch's `Country` column.
# A timeline-only name with no metadata row would vanish from a by-country
# view, so it gets a country here. Empty today — every timeline name resolves;
# test_country_fallback_only_names_sites_epoch_left_blank retires an entry as
# soon as Epoch catalogues the site.
_DC_COUNTRY_FALLBACK = {}
_DC_CTY_US = "United States"
_DC_CTY_CN = "China"
# Sites outside China that Chinese labs train on. Epoch's own source notes on
# DayOne Nusajaya cite the FT reporting Alibaba and ByteDance training models
# in Southeast Asia, and DayOne is the GDS Holdings spin-off. Counting the
# whole campus as Chinese-accessible is an upper-bound reading — DayOne has
# other tenants — so it is a selector, and its label says what it assumes.
_DC_CN_ACCESS_ABROAD = ("DayOne Nusajaya",)
_DC_CTY_CN_ACCESS = "China-accessible"
# Extrapolation: log-linear OLS on monthly samples of a country's step series
# since this year (the tab's earliest chart start), pace uncertainty as the
# larger of the fit's standard error, the spread of the slope across the
# lookback windows, and a floor — the floor is what carries the cone for a
# country with two years of data, where the windows all coincide.
_DC_CTY_FIT_SINCE = 2023
_DC_CTY_FIT_WINDOWS = (2023, 2024, 2025, 2026)
_DC_CTY_SIGMA_G_FLOOR = 0.10      # OOM/yr, 1σ
_DC_CTY_MIN_FIT_POINTS = 6        # monthly samples; fewer → borrow the US pace
_DC_CTY_SINCE_YEARS = [2023, 2024, 2025, 2026]
# Timing noise on planned steps: a future step's date moves by a Normal(0, σ)
# fraction of its lead time (how far past today it sits) — symmetric, so
# Epoch's dates centre the band; Epoch dates conservatively (it already pushes
# doubtful completions out), so a one-sided lateness model was tried and put
# the plan at the top edge of the interval instead. σ depends on how much of
# the country's planned buildout Epoch documents from a schedule, filing or
# statement (_DC_PLAN_SOURCED_RE on `Construction status`) rather than
# estimates by analogy; it interpolates between the two values by that share.
# A heuristic on prose — Epoch publishes no confidence column.
_DC_CTY_SLIP_SIGMA_Q = {'sourced': 0.15, 'estimate': 0.35}  # σ, fraction of lead
# How far past today the catalogue is treated as complete. Beyond it the trend
# takes over and known plans are only a floor: Epoch's list thins out with
# distance (one site dated 2030), and anchoring the trend on the last entry
# held the US line flat through 2029.
_DC_CTY_PLAN_HORIZON_DAYS = 548
# Level uncertainty on a plan, OOM per year of lead (1σ, symmetric): a site
# can come in under or over its stated size, and plans a year out are less
# exact than next quarter's. Small on purpose; slip carries the downside.
_DC_CTY_PLAN_LEVEL_SIGMA = 0.06
_DC_PLAN_SOURCED_RE = re.compile(
    r'\]\(http|\bschedul|\bfiling|\bstated\b|\bpermit|\bannounc|\bpress release',
    re.I)
_DC_CTY_CN_DOMESTIC = "China (domestic only)"
_DC_CTY_PACE_OPTIONS = {
    "The US trend for every country (a follower tracks the leader)": 'us',
    "Each country's own fitted trend": 'own',
}
_DC_CTY_COLORS = {_DC_CTY_US: "#1F77B4", _DC_CTY_CN: "#D62728",
                  _DC_CTY_CN_ACCESS: "#D62728", _DC_CTY_CN_DOMESTIC: "#E07B00"}
# Reference countries draw from a palette with no reds or blues, so nothing
# reads as the US or China.
_DC_CTY_OTHER_COLORS = ("#2CA02C", "#9467BD", "#8C564B", "#7F7F7F", "#17BECF",
                        "#BCBD22", "#E377C2", "#4B6B3A")

# ── Share of catalogued capacity by region ───────────────────────────────
# Buckets for the share chart at the bottom of the tab, in stack order, keyed
# on Epoch's `Country` column. Geography only: a site counts where it stands,
# whoever trains in it — so DayOne Johor sits in SEA here while the country
# panel above also reads it as China-accessible. Anything unmatched, and any
# site Epoch leaves without a country, falls to the residual bucket.
_DC_REGION_OTHER = "Other"
_DC_REGION_SEA = "SEA"
_DC_REGION_EU = "Europe/UK"
_DC_REGIONS = (
    ("US domestic", frozenset({_DC_CTY_US})),
    ("China domestic", frozenset({_DC_CTY_CN})),
    (_DC_REGION_SEA, frozenset({
        "Malaysia", "Indonesia", "Singapore", "Thailand", "Vietnam",
        "Philippines", "Brunei", "Cambodia", "Laos", "Myanmar"})),
    # The EU, the UK and the rest of Europe — the non-EU hosts (Norway today)
    # belong with them, not in the residual bucket.
    (_DC_REGION_EU, frozenset({
        "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany",
        "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
        "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania",
        "Slovakia", "Slovenia", "Spain", "Sweden", "United Kingdom",
        "Iceland", "Liechtenstein", "Norway", "Switzerland"})),
    ("UAE", frozenset({"United Arab Emirates"})),
)
_DC_REGION_COLORS = {
    "US domestic": "#1F77B4", "China domestic": "#D62728",
    _DC_REGION_SEA: "#F06BA8", _DC_REGION_EU: "#9467BD", "UAE": "#16A085",
    _DC_REGION_OTHER: "#999999",
}

_DC_EXCLUDE_COMPANIES = {
    "QTS", "DayOne", "CoreWeave", "STACK", "Stream", "Vantage", "EdgeCore",
    "Oracle", "Microsoft",
}

# …but only while they stay small. A listed company whose biggest site reaches
# this within _DC_EXCLUDE_HORIZON_DAYS is charted anyway. See
# _dc_hidden_companies() for why the test is framed this way.
_DC_EXCLUDE_MIN_H100 = 100_000
_DC_EXCLUDE_HORIZON_DAYS = 365


def _dc_company_peak_h100(dcs, cap_date=None):
    """company → its largest single site's peak H100-equivalents.

    Points dated after `cap_date` are ignored when one is given.
    """
    peak = {}
    for dc in dcs:
        vals = [p['h100'] for p in dc['points']
                if p.get('h100') is not None
                and (cap_date is None or p['date'] <= cap_date)]
        if vals:
            peak[dc['company']] = max(peak.get(dc['company'], 0.0), max(vals))
    return peak


def _dc_hidden_companies(dcs, now=None):
    """The subset of _DC_EXCLUDE_COMPANIES small enough to leave off the tab.

    The exclude list names companies that aren't AI labs — colocation and
    neutral-host operators whose recorded "company" is a landlord rather than
    whoever trains on the hardware. Hiding them unconditionally was fine only
    while they were small, and they are not: QTS Cedar Rapids is the single
    largest site in Epoch's data, so dropping it made "Largest single data
    center" understate the frontier and name a smaller site as record holder.
    Size now gates the exclusion — a listed company disappears only while its
    biggest site stays under _DC_EXCLUDE_MIN_H100.

    Three things about how the test is framed:

    1. **H100-equivalents, always** — never the metric currently selected. The
       same number in megawatts or dollars means something else entirely, and
       two of the tab's metrics are inverted (bigger site = smaller value), so a
       ">= threshold" test would read backwards on them.
    2. **A rolling `_DC_EXCLUDE_HORIZON_DAYS` horizon, not the uncapped peak.**
       Uncapped, every listed company eventually clears 100k on buildout
       announced for 2028 and later, and the exclusion stops meaning anything.
       A year out is the span where the plans are firm enough to chart.
       The roster now moves as that horizon rolls, and the movement is the
       catalogue densifying rather than a refresh: Vantage qualified on
       2026-09-01 on data unchanged from the week before, when its planned
       154k-H100e TX1 step (2027-08-31) came inside the year. STACK (250k)
       is next, about six months out. So expect
       `test_current_roster_is_what_the_tab_says_it_is` to fail roughly
       whenever a new site's plan crosses the line, and retarget it
       deliberately — the alternatives are a higher bar or a shorter horizon,
       and both only postpone the same call.
    3. **Not the user's projection window.** dc_end_year caps the *charts*, but
       who appears on the tab is a property of the company, not of a slider —
       a roster that changed with the controls would read as sites blinking in
       and out of existence.

    Companies not on the exclude list are unaffected; this only ever un-hides.
    """
    cap = (now or datetime.now()) + timedelta(days=_DC_EXCLUDE_HORIZON_DAYS)
    peak = _dc_company_peak_h100(dcs, cap_date=cap)
    return {co for co in _DC_EXCLUDE_COMPANIES
            if peak.get(co, 0.0) < _DC_EXCLUDE_MIN_H100}


def _dc_unattributed_companies(dcs):
    """Companies whose every site's label is a guess from the building's name.

    Epoch records no user and no owner for these, so load_data_centers() falls
    back to the first token of the site name: the capacity is real but the
    tenant is unknown, and "QTS" on a chart means a landlord, not an AI lab
    running 3.7M H100-equivalents. Rendered with a † and a footnote wherever the
    company is named. A company with even one properly attributed site is left
    alone — Microsoft is listed as its own sites' user, so it is not marked.
    """
    seen, attributed = set(), set()
    for dc in dcs:
        seen.add(dc['company'])
        if dc.get('attributed'):
            attributed.add(dc['company'])
    return seen - attributed


_DC_UNATTRIBUTED_MARK = " †"


def _dc_co_label(co, unattributed):
    """Company name for display, marked when nobody is recorded as using it."""
    return co + _DC_UNATTRIBUTED_MARK if co in unattributed else co


def _fmt_duration_days(days):
    """A duration in days as a rough human-readable span ("~3 weeks").

    Deliberately coarse — these come from order-of-magnitude FLOP estimates, so
    the unit is picked to keep the number small and the "~" is always shown.
    """
    if days is None or not np.isfinite(days) or days <= 0:
        return "—"

    def _u(v, unit):
        s = f"{v:.1f}".rstrip('0').rstrip('.') if v < 10 else f"{v:,.0f}"
        return f"~{s} {unit}{'' if s == '1' else 's'}"

    if days < 1 / 24:
        mins = days * 24 * 60
        return f"~{mins:.0f} min" if mins >= 1 else "<1 min"
    if days < 1:
        return _u(days * 24, "hour")
    if days < 7:
        return _u(days, "day")
    # Each unit switches only at a whole 1 of the next one up, so a value is
    # never reported as a fraction ("~0.9 months").
    if days < 30.4375:
        return _u(days / 7, "week")
    if days < 365.25:
        return _u(days / 30.4375, "month")
    return _u(days / 365.25, "year")


def _logop_num(lf):
    """A log₁₀ operation count to one decimal, integers kept bare: 28.0 → "28"."""
    if lf is None or not np.isfinite(lf):
        return "—"
    return f"{lf:.1f}".rstrip('0').rstrip('.')


def _logop_lbl(lf):
    """A log₁₀ operation count with its unit, for prose and hovers."""
    n = _logop_num(lf)
    return n if n == "—" else f"{n} log OP"


def _log_op(v):
    """log₁₀ of a raw operation count, to one decimal with a bare integer kept
    bare: 1e28 → "28", 2e28 → "28.3"."""
    if v is None or not np.isfinite(v) or v <= 0:
        return "—"
    return _logop_num(np.log10(v))


def _and_list(items):
    """['a','b','c'] → "a, b and c" — for naming companies in prose captions."""
    items = list(items)
    if len(items) <= 1:
        return items[0] if items else ""
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _dc_fmt_value(v, kind):
    if v is None:
        return "—"
    if kind == "h100":
        if v >= 1e6:
            return f"{v / 1e6:.2f}M"
        if v >= 1e3:
            return f"{v / 1e3:.0f}k"
        return f"{v:.0f}"
    if kind == "mw":
        return f"{v:,.0f} MW"
    if kind == "cost":
        return f"${v:.1f}B"
    if kind == "sci":
        return f"{v:.2e}"
    if kind == "flop":
        # Reported as log₁₀ of the operation count (1e28 → 28, 2e28 → 28.3);
        # the stored value stays raw so sums and maxima keep working.
        return f"{_log_op(v)} log OP" if v > 0 else "—"
    if kind == "traintime":
        # v is training runs per 2-month window; report the time for one run.
        return _fmt_duration_days(_DAYS_2MO / v) if v > 0 else "—"
    return f"{v:g}"


_DC_PARTY_OPTIONS = {"Tenant (who trains there)": 'tenant',
                     "Operator (who owns the building)": 'operator'}


def _dc_with_party(dcs, party):
    """The site list under one attribution, with a `companies` membership list
    the per-company aggregators group on.

    'tenant' keeps the single `company` label (first-listed user → owner →
    name token) but sets `companies` to **every** listed user, so a shared
    site counts under each of its tenants — Colossus 2 under Anthropic,
    Cursor and SpaceXAI alike, since all three train there. Per-company lines
    are capability views (max per company), so this never double-counts
    within a line, but lines aren't additive across companies. 'operator'
    credits each building to its owner alone — Colossus to SpaceXAI,
    Stargate Abilene to Oracle."""
    if party == 'operator':
        return [dict(dc, company=dc['operator'], companies=[dc['operator']])
                for dc in dcs]
    return [dict(dc, companies=dc['users'] or [dc['tenant']]) for dc in dcs]


def _dc_series_for_metric(dcs, key, cap_date=None):
    """Per-DC step points for one metric: name → {company, pts:[(date, val)…]}.

    Drops missing values and any points after cap_date (when given).
    """
    out = {}
    for dc in dcs:
        pts = []
        for p in dc['points']:
            v = p.get(key)
            if v is None:
                continue
            if cap_date is not None and p['date'] > cap_date:
                continue
            pts.append((p['date'], v))
        if pts:
            out[dc['name']] = {'company': dc['company'], 'pts': pts,
                               'companies': dc.get('companies',
                                                   [dc['company']])}
    return out


def _dc_val_at(pts, t):
    """Forward-filled value at time t: the latest point with date <= t (or None)."""
    dates = [d for d, _ in pts]
    i = bisect.bisect_right(dates, t) - 1
    return pts[i][1] if i >= 0 else None


def _dc_envelope(series):
    """Largest single data center at each event date.

    Returns [(date, value, leader_name, leader_company), …] over the union of all
    event dates, picking the max forward-filled value across every data center.
    """
    all_dates = sorted({d for v in series.values() for d, _ in v['pts']})
    steps = []
    for d in all_dates:
        best = None
        best_name = None
        best_co = None
        for name, v in series.items():
            val = _dc_val_at(v['pts'], d)
            if val is None:
                continue
            if best is None or val > best:
                best, best_name, best_co = val, name, v['company']
        if best is not None:
            steps.append((d, best, best_name, best_co))
    return steps


def _dc_company_series(series):
    """Per company, the largest of its data centers at each event date.

    Returns company → [(date, value, leader_dc_name), …].
    """
    companies = {}
    for name, v in series.items():
        for co in v.get('companies', [v['company']]):
            companies.setdefault(co, []).append((name, v['pts']))
    out = {}
    for co, members in companies.items():
        all_dates = sorted({d for _, pts in members for d, _ in pts})
        steps = []
        for d in all_dates:
            best = None
            best_name = None
            for name, pts in members:
                val = _dc_val_at(pts, d)
                if val is None:
                    continue
                if best is None or val > best:
                    best, best_name = val, name
            if best is not None:
                steps.append((d, best, best_name))
        out[co] = steps
    return out


def _dc_network_site_clusters(level='fabric'):
    """site name → cluster label, for sites that can share one training job.

    Sites absent from the map are their own cluster, so nothing pools with them.
    `level` is the weakest basis admitted, in increasing order of speculation
    (_DC_NETWORK_LEVELS): 'proximity' keeps only sites physically near one
    another, 'fabric' adds the purpose-built long-haul links someone has
    announced, 'plausible' adds regional groups nobody has announced.

    Bases are applied weakest-first and a later one overwrites the sites it
    names, so a regional group subsumes the metro clusters inside it. That is a
    merge rather than a re-cut only because every 'plausible' entry contains any
    lower cluster it touches whole; TestDcNetworkClusters holds that invariant,
    which is what makes each level a superset of the one before it.
    """
    keep = _DC_NETWORK_LEVELS[:_DC_NETWORK_LEVELS.index(level) + 1]
    out = {}
    for basis in keep:
        for label, b, names in _DC_NETWORK_CLUSTERS:
            if b != basis:
                continue
            for name in names:
                out[name] = label
    return out


def _dc_company_networked_series(series, cluster_of):
    """Per company, its largest *networkable group* of sites at each event date.

    The multi-site view of _dc_company_series(): instead of one site, sum the
    sites a company could plausibly drive as a single training job, then take
    its biggest such group. Grouping is (company, cluster), so a site with no
    cluster stands alone and `cluster_of={}` reduces exactly to
    _dc_company_series(). `cluster_of=None` pools every site a company has —
    the upper bound, kept only for comparison.

    Summing is valid for every metric the tab offers, the 'traintime' ones
    included: those are stored as training runs per 2-month window (see
    load_data_centers), so two equal sites are twice the runs, i.e. half the
    time to train one model. Storing the count is what keeps this a plain sum.

    Returns company → [(date, value, site_names, cluster_label), …] with
    site_names largest first and cluster_label None for a lone site.
    """
    companies = {}
    for name, v in series.items():
        key = None if cluster_of is None else cluster_of.get(name, name)
        for co in v.get('companies', [v['company']]):
            companies.setdefault(co, []).append((name, key, v['pts']))
    out = {}
    for co, members in companies.items():
        all_dates = sorted({d for _, _, pts in members for d, _ in pts})
        steps = []
        for d in all_dates:
            groups = {}
            for name, key, pts in members:
                val = _dc_val_at(pts, d)
                if val is not None:
                    groups.setdefault(key, []).append((val, name))
            if not groups:
                continue
            best_key, best_total, best_vals = None, None, None
            for gkey, vals in groups.items():
                total = sum(v for v, _ in vals)
                if best_total is None or total > best_total:
                    best_key, best_total, best_vals = gkey, total, vals
            best_vals = sorted(best_vals, reverse=True)
            if len(best_vals) == 1:
                label = None                      # a lone site, not a cluster
            elif best_key is None:
                label = "all sites"               # the unrestricted upper bound
            else:
                label = best_key
            steps.append((d, best_total,
                          tuple(n for _, n in best_vals), label))
        out[co] = steps
    return out


def _dc_color(company, idx):
    return _DC_COLORS.get(company, _DC_PALETTE[idx % len(_DC_PALETTE)])


def _dc_site_country(dc):
    """A site's country: Epoch's column, else the curated fallback, else ''."""
    return dc.get('country') or _DC_COUNTRY_FALLBACK.get(dc['name'], '')


def _dc_country_groups(series, country_of, cn_scope='abroad'):
    """country label → [site names], for the by-country panel.

    `cn_scope='abroad'` moves the sites in _DC_CN_ACCESS_ABROAD out of their
    own country into a _DC_CTY_CN_ACCESS group alongside every site in China,
    so nothing is counted twice; 'domestic' leaves them where they stand and
    labels the Chinese group plainly. Sites with no country are dropped.
    """
    groups = {}
    for name in series:
        c = country_of.get(name, '')
        if cn_scope == 'abroad' and (c == _DC_CTY_CN or name in _DC_CN_ACCESS_ABROAD):
            c = _DC_CTY_CN_ACCESS
        if not c:
            continue
        groups.setdefault(c, []).append(name)
    return groups


def _dc_region_of(country):
    """Which _DC_REGIONS bucket a country falls in; the residual otherwise."""
    for label, members in _DC_REGIONS:
        if country in members:
            return label
    return _DC_REGION_OTHER


def _dc_region_totals(series, country_of, grid):
    """Region label → the metric summed over its sites at each grid date.

    Every site's forward-filled value, added into the bucket its country falls
    in; a region with nothing anywhere on the grid is dropped. Values stay in
    the metric's own units, so the caller normalises. Only additive metrics
    mean anything here — every _DC_METRICS key is one, the two "time to
    train" ones because they store runs per window (see _dc_share_label).
    """
    out = {}
    for name, v in series.items():
        vals = out.setdefault(_dc_region_of(country_of.get(name, '')),
                              [0.0] * len(grid))
        for i, d in enumerate(grid):
            x = _dc_val_at(v['pts'], d)
            if x:
                vals[i] += x
    order = [lab for lab, _ in _DC_REGIONS] + [_DC_REGION_OTHER]
    return {lab: out[lab] for lab in order
            if lab in out and any(x > 0 for x in out[lab])}


def _dc_region_shares(totals, grid):
    """(grid, {region: [percent]}, {region: [level]}), all trimmed to the
    dates where something is built — a share of nothing is not 0%, it is
    undefined. Levels come back trimmed alongside so a caller can hover the
    raw value against the share without re-indexing.
    """
    keep = [i for i in range(len(grid))
            if sum(v[i] for v in totals.values()) > 0]
    if not keep:
        return [], {}, {}
    g = [grid[i] for i in keep]
    shares, levels = {}, {}
    for lab, vals in totals.items():
        levels[lab] = [vals[i] for i in keep]
        shares[lab] = [100.0 * vals[i] / sum(v[i] for v in totals.values())
                       for i in keep]
    return g, shares, levels


def _dc_share_label(metric_label, kind):
    """What a share of this metric is a share of. The two "time to train"
    metrics store runs per 2-month window — additive, and what a compute
    share means — so they are named that way rather than by their inverted
    label, which no share can be taken of."""
    if kind == 'traintime':
        return "training runs per 2-month window"
    return metric_label


def _dc_country_steps(series, names, mode, cluster_of):
    """One country's capacity over time as [(date, value, detail), …].

    mode 'site'    — its largest single site (detail = site name)
    mode 'company' — the largest networkable group any one company there has,
                     pooled exactly as _dc_company_networked_series() does it
                     (detail = "company — cluster" or the site)
    mode 'country' — every site in the country summed (detail = site count):
                     the state-direction claim the Pacing tab makes for China,
                     an upper bound for anyone else
    """
    sub = {n: series[n] for n in names if n in series}
    if not sub:
        return []
    if mode == 'site':
        return [(d, v, n) for d, v, n, _ in _dc_envelope(sub)]
    if mode == 'company':
        per_co = _dc_company_networked_series(sub, cluster_of)
        dates = sorted({d for steps in per_co.values() for d, *_ in steps})
        out = []
        for d in dates:
            best = None
            for co, steps in per_co.items():
                val = _dc_val_at([(s[0], s[1]) for s in steps], d)
                if val is None:
                    continue
                if best is None or val > best[0]:
                    last = next(s for s in reversed(steps) if s[0] <= d)
                    best = (val, f"{co} — {last[3]}" if last[3] else
                            f"{co} — {last[2][0]}")
            if best is not None:
                out.append((d, best[0], best[1]))
        return out
    dates = sorted({d for v in sub.values() for d, _ in v['pts']})
    out = []
    for d in dates:
        vals = [x for x in (_dc_val_at(v['pts'], d) for v in sub.values())
                if x is not None]
        out.append((d, sum(vals), f"{len(vals)} sites"))
    return out


def _dc_plan_quality(dcs, names, today):
    """Share of a site group's future capacity rows whose construction status
    cites a document (see _DC_PLAN_SOURCED_RE). None when nothing is planned."""
    fut = [p for dc in dcs if dc['name'] in names
           for p in dc['points'] if p['date'] > today]
    if not fut:
        return None
    return sum(bool(_DC_PLAN_SOURCED_RE.search(p.get('status') or ''))
               for p in fut) / len(fut)


def _dc_cty_slip_sigma(quality):
    """Timing-noise σ, as a fraction of lead time, for a plan of this quality
    (share of rows sourced); the estimate value when quality is unknown."""
    if quality is None:
        return _DC_CTY_SLIP_SIGMA_Q['estimate']
    return (_DC_CTY_SLIP_SIGMA_Q['estimate']
            + quality * (_DC_CTY_SLIP_SIGMA_Q['sourced'] - _DC_CTY_SLIP_SIGMA_Q['estimate']))


def _dc_cty_month_grid(start, end):
    """First-of-month datetimes from `start`'s month through `end`'s."""
    out = []
    y, m = start.year, start.month
    while datetime(y, m, 1) <= end:
        out.append(datetime(y, m, 1))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _dc_cty_fit(steps, since=None, t_end=None):
    """Log-linear OLS of a step series, sampled monthly up to its last change.

    Returns {t0, v0, g, se, sigma_g, sigma_res, n, windows} — t0/v0 the anchor
    (last recorded step), g the central pace in OOM/yr fitted on samples from
    `since` (default _DC_CTY_FIT_SINCE) on, windows the pace over every
    lookback in _DC_CTY_FIT_WINDOWS that has enough points whatever `since`
    is, sigma_g the 1σ pace uncertainty = max(se, window spread / 2·1.28,
    floor) — so the windows the user did not pick still widen the cone. None when fewer than
    _DC_CTY_MIN_FIT_POINTS positive monthly samples exist or the anchor is not
    positive.
    """
    if not steps:
        return None
    pts = [(s[0], s[1]) for s in steps]
    # Anchor: the last recorded step, or the value carried at `t_end` when the
    # catalogue runs past it (its tail is treated as a floor, not an anchor).
    if t_end is not None and t_end < steps[-1][0]:
        t0, v0 = t_end, _dc_val_at(pts, t_end)
    else:
        t0, v0 = steps[-1][0], steps[-1][1]
    if v0 is None or v0 <= 0:
        return None
    first_pos = next((d for d, v in pts if v is not None and v > 0), None)
    if first_pos is None:
        return None
    since = datetime(since or _DC_CTY_FIT_SINCE, 1, 1)
    grid = _dc_cty_month_grid(max(first_pos, datetime(_DC_CTY_FIT_WINDOWS[0], 1, 1)), t0)
    all_samples = [(d, _dc_val_at(pts, d)) for d in grid]
    all_samples = [(d, v) for d, v in all_samples if v is not None and v > 0]
    # The anchor itself, so the fit sees the latest step even mid-month.
    all_samples.append((t0, v0))
    samples = [x for x in all_samples if x[0] >= since]
    if len(samples) < _DC_CTY_MIN_FIT_POINTS:
        return None

    def _ols(sub):
        t = np.array([(d - t0).days / 365.25 for d, _ in sub])
        y = np.log10([v for _, v in sub])
        a, b = fit_line(t, y)
        resid = y - (a + b * t)
        dof = len(sub) - 2
        sxx = float(((t - t.mean()) ** 2).sum())
        se = (float(np.sqrt(resid @ resid / dof / sxx))
              if dof > 0 and sxx > 0 else 0.0)
        return float(b), se, float(resid.std())

    g, se, sigma_res = _ols(samples)
    windows = {}
    for yr in _DC_CTY_FIT_WINDOWS:
        sub = [s for s in all_samples if s[0] >= datetime(yr, 1, 1)]
        if len(sub) >= _DC_CTY_MIN_FIT_POINTS:
            windows[yr] = _ols(sub)[0]
    spread = ((max(windows.values()) - min(windows.values())) / (2 * 1.282)
              if windows else 0.0)
    sigma_g = max(se, spread, _DC_CTY_SIGMA_G_FLOOR)
    return {'t0': t0, 'v0': v0, 'g': g, 'se': se, 'sigma_g': sigma_g,
            'sigma_res': sigma_res, 'n': len(samples), 'windows': windows}


def _dc_cty_trajectories(steps, fit, grid, n, pace=None, today=None,
                         slip_sigma=None):
    """(n, len(grid)) array of sampled capacity paths.

    Up to `today` every sample is the recorded step value. Past today, planned
    steps move in time: sample i reads the plan at d − (d − today)·f_i with
    f_i ~ Normal(0, `slip_sigma`) — symmetric, so the plan centres the band,
    which opens at today and widens with lead. With `today` or `slip_sigma`
    None nothing moves. Past t0 each sample extrapolates from
    its own realized value there: log10 v = log10 v(t0) + g·τ + ε·min(τ, 1yr),
    with g ~ N(pace.g, pace.sigma_g) and ε ~ N(0, fit.sigma_res), floored at
    the slipped plan where the catalogue runs past t0. Inside the plan window
    each sample also carries a level draw of _DC_CTY_PLAN_LEVEL_SIGMA OOM per
    year of lead, so a flat plan still has a (small) band. Every path is made
    non-decreasing at the end — a known site still
    comes online; the trend adds what isn't catalogued yet. `pace` defaults
    to `fit` and can be another country's, for a borrowed trend. NaN where
    nothing is recorded and no fit extends it.
    """
    pts = [(s[0], s[1]) for s in steps]
    out = np.full((n, len(grid)), np.nan)
    if not pts:
        return out
    last = pts[-1][0]
    days = np.array([(d - last).days for d, _ in pts], dtype=float)
    vals = np.array([np.nan if v is None else v for _, v in pts], dtype=float)
    slip = (np.random.normal(0.0, slip_sigma, n)
            if today is not None and slip_sigma else None)
    level = np.random.normal(0.0, _DC_CTY_PLAN_LEVEL_SIGMA, n)

    def _realized(d):
        """Per-sample value of the recorded step series at date d: past today
        each sample reads the plan at its slipped date and scales it by its
        level draw, so a step lands late by a sample-specific amount and a flat
        stretch still carries a (small) band."""
        q = np.full(n, float((d - last).days))
        scale = 1.0
        if slip is not None and d > today:
            lead = (d - today).days
            # A slipped plan cannot fall below what is already built today.
            q = np.maximum(q - lead * slip, float((today - last).days))
            scale = 10 ** (level * lead / 365.25)
        idx = np.searchsorted(days, q, side='right') - 1
        v = np.where(idx >= 0, vals[np.clip(idx, 0, len(vals) - 1)], np.nan)
        return v * scale

    for j, d in enumerate(grid):
        if d <= last:
            out[:, j] = _realized(d)
    def _monotone(a):
        """Built capacity does not go away: each path non-decreasing, with
        NaN kept where nothing was recorded or extrapolated."""
        return np.where(np.isnan(a), np.nan, np.fmax.accumulate(a, axis=1))

    if fit is None:
        return _monotone(out)
    pace = pace or fit
    g = np.random.normal(pace['g'], pace['sigma_g'], n)
    eps = np.random.normal(0.0, fit['sigma_res'], n)
    anchor = _realized(fit['t0'])
    anchor = np.where(np.isnan(anchor) | (anchor <= 0), fit['v0'], anchor)
    base = np.log10(anchor)
    for j, d in enumerate(grid):
        if d <= fit['t0']:
            continue
        tau = (d - fit['t0']).days / 365.25
        trend = 10 ** (base + g * tau + eps * min(tau, 1.0))
        # The slipped plan is a floor, carried forward past its last entry.
        floor = out[:, j] if d <= last else _realized(last)
        out[:, j] = np.fmax(trend, floor)
    return _monotone(out)


def _dc_cty_lag_months(follower, leader, grid):
    """Months by which `follower` trails `leader`, per sample and grid date:
    the gap between a date and the first date the leader's running max reached
    the follower's value then. Negative means the follower is ahead; where the
    leader never gets there within the grid the lag is floored at one month
    past the grid's end — a bound, so the percentiles stay honest rather than
    dropping exactly the samples where the follower leads. NaN where either
    side is unrecorded. Returns (lag, unresolved), the second a boolean mask
    of the floored cells.
    """
    n, m = follower.shape
    lead_max = np.fmax.accumulate(np.where(np.isnan(leader), -np.inf, leader),
                                  axis=1)
    months = np.array([(d.year - grid[0].year) * 12 + d.month - grid[0].month
                       for d in grid], dtype=float)
    out = np.full((n, m), np.nan)
    unresolved = np.zeros((n, m), dtype=bool)
    for j in range(m):
        target = follower[:, j][:, None]
        hit = lead_max >= target                       # (n, m)
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        known = ~np.isnan(follower[:, j])
        ok = any_hit & known
        out[ok, j] = months[j] - months[first[ok]]
        out[known & ~any_hit, j] = months[j] - (months[-1] + 1)
        unresolved[:, j] = known & ~any_hit
    return out, unresolved


# ── Load data (before sidebar, so model names are available) ─────────────

metr_all = load_metr_all(_mtime=_yaml_mtime())
frontier_all = [m for m in metr_all if m['is_sota']]
gpt4o_idx = next(i for i, m in enumerate(frontier_all) if m['name'] == 'gpt_4o_inspect')
frontier_names = [pretty(m['name']) for m in frontier_all]

eci_all = load_eci_frontier(_mtime=_eci_mtime())
eci_frontier_all = [m for m in eci_all if m['is_frontier']]
eci_frontier_names = [m['display_name'] for m in eci_frontier_all]

rli_all = load_rli_data()
rli_frontier_all = [m for m in rli_all if m['is_frontier']]
rli_frontier_names = [m['name'] for m in rli_frontier_all]

rsi_all = load_rsi_data()
rsi_frontier_all = [m for m in rsi_all if m['is_frontier']]
rsi_dir_all = load_rsi_direction()
rsi_dir_frontier_all = [r for r in rsi_dir_all if r['is_frontier']]

dc_all = load_data_centers(_mtime=_dc_mtime())

ukc_all = load_ukcyber(_mtime=_ukc_mtime())
ukc_frontier_all = [m for m in ukc_all if m['is_frontier']]
ukc_frontier_names = [m['name'] for m in ukc_frontier_all]


# ── Sidebar: tab selector ────────────────────────────────────────────────

_TAB_OPTIONS = ["METR Horizon", "Epoch ECI", "ECI Company Gap", "Remote Labor Index", "RSI", "UK Cyber", "Employment", "Revenue", "Data Centers", "Compute/capabilities/diffusion", "Pacing"]
_SLUG_FOR_TAB = {"METR Horizon": "metr", "Epoch ECI": "eci", "Remote Labor Index": "rli", "RSI": "rsi", "UK Cyber": "ukcyber", "Revenue": "revenue", "Employment": "employment", "ECI Company Gap": "ecigap", "Data Centers": "datacenters", "Compute/capabilities/diffusion": "computecap", "Pacing": "pacing"}
_TAB_SLUG = {_SLUG_FOR_TAB[t]: i for i, t in enumerate(_TAB_OPTIONS)}

# Read ?tab= from URL for deep-linking
_url_tab = st.query_params.get("tab", "").lower()
_default_tab_idx = _TAB_SLUG.get(_url_tab, 0)

with st.sidebar:
    active_tab = st.radio("Tab", _TAB_OPTIONS, index=_default_tab_idx, horizontal=True, key="_active_tab")
    st.markdown("---")

# Keep URL in sync with selected tab (omit when at default)
_DEFAULT_TAB = _TAB_OPTIONS[0]
if active_tab == _DEFAULT_TAB:
    if "tab" in st.query_params:
        del st.query_params["tab"]
else:
    st.query_params["tab"] = _SLUG_FOR_TAB[active_tab]

# a/b (and legacy region) are ECI-only params; drop them on any other tab
if active_tab != "Epoch ECI":
    for _p in ("a", "b", "region"):
        if _p in st.query_params:
            del st.query_params[_p]


# ── Section deep links (?to=<anchor slug>) ───────────────────────────────
#
# Streamlit gives every heading an `id`, but a `#heading` link cannot reach it
# on a fresh load: Community Cloud serves the app in an iframe whose src copies
# the query string and drops the fragment, and every `st.query_params` write
# rebuilds the URL from path + search alone, dropping it again. `?to=` rides
# the query string instead. It is consumed on arrival so later reruns don't
# jump the page back, and `_render_anchor_links()` rewrites each heading's own
# link icon to emit this form.

_ANCHOR_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,120}$")


def _anchor_slug(raw):
    """`raw` if it looks like a Streamlit heading anchor, else ''.

    Whitelisting the shape is what lets the slug be inlined into the injected
    script as a bare literal.
    """
    s = str(raw).strip().lstrip("#").lower()
    return s if _ANCHOR_SLUG_RE.match(s) else ""


_url_to = _anchor_slug(st.query_params.get("to", ""))
if "to" in st.query_params:
    del st.query_params["to"]



# ── METR Horizon ─────────────────────────────────────────────────────────

_METR_RESET_KEYS = [
    "custom_dt_lo", "custom_dt_hi",
    "custom_pos_lo_p50", "custom_pos_lo_p80",
    "custom_pos_hi_p50", "custom_pos_hi_p80",
    "piecewise_n_seg", "bp1_select", "bp2_select",
    "custom_dt_dist", "custom_pos_dist",
    "superexp_dt_init", "superexp_halflife",
    "superexp_dt_floor", "superexp_dt_ci_lo",
    "superexp_dt_ci_hi", "superexp_pos_lo_p50",
    "superexp_pos_lo_p80", "superexp_pos_hi_p50",
    "superexp_pos_hi_p80",
    "metr_proj_basis",
    "milestones", "labels", "post_gpt4o", "p80",
    "log_scale", "hours_only", "_proj_as_of", "metr_end_year",
    "_metr_seg_config",
]

_METR_DEFAULTS = {
    "metr_proj_basis": "Piecewise linear",
    "piecewise_n_seg": 2,
    "custom_dt_dist": "Lognormal",
    "custom_pos_dist": "Lognormal",
    "milestones": True,
    "labels": True,
    "post_gpt4o": False,
    "p80": False,
    "log_scale": True,
    "hours_only": False,
    "metr_end_year": 2026,
}

def render_metr():
    if st.session_state.pop("_reset_metr", False):
        for k in _METR_RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(_METR_DEFAULTS)
        st.rerun()

    # Initialize widget defaults on first run (no explicit index=/value= on widgets)
    for k, v in _METR_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── METR Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.header("METR Projection")

        # Read "project as of" from session state (widget rendered at bottom of sidebar)
        proj_as_of_name = st.session_state.get('_proj_as_of', frontier_names[-1])
        if proj_as_of_name not in frontier_names:
            proj_as_of_name = frontier_names[-1]
        proj_as_of_idx = frontier_names.index(proj_as_of_name)

        # --- Projection basis ---
        basis_options = ["Linear", "Piecewise linear", "Superexponential"]
        proj_basis = st.radio("Projection basis", basis_options, key="metr_proj_basis")

        # Read p80 toggle from session state (widget rendered below, but state persists)
        _sidebar_p80 = st.session_state.get('p80', False)
        _sb_val_key = 'p80_min' if _sidebar_p80 else 'p50_min'
        _sb_lo_key = 'p80_lo' if _sidebar_p80 else 'p50_lo'
        _sb_hi_key = 'p80_hi' if _sidebar_p80 else 'p50_hi'

        custom_pos_lo = custom_pos_hi = custom_dt_lo = custom_dt_hi = None
        custom_dt_dist = "Lognormal"
        custom_pos_dist = "Lognormal"
        piecewise_n_segments = 1
        piecewise_breakpoints = []
        _is_linear = proj_basis in ("Linear", "Piecewise linear")
        if proj_basis == "Piecewise linear":
            piecewise_n_segments = 2  # default for piecewise

        # Pre-compute OLS DT for data-driven defaults
        _pre_fr = frontier_all[:proj_as_of_idx + 1]
        _pre_base = frontier_all[0]['date']
        _pre_days = np.array([(m['date'] - _pre_base).days for m in _pre_fr], dtype=float)
        _pre_vals = np.array([np.log2(m['p50_min']) for m in _pre_fr])
        _pre_params = fit_line(_pre_days, _pre_vals)
        _pre_ols_dt = round(1.0 / _pre_params[1]) if _pre_params[1] > 0 else 100

        if _is_linear:
            with st.expander("Advanced options"):
                st.button("Reset to defaults", key="reset_linear",
                          on_click=lambda: st.session_state.update(_reset_metr=True))

                # Segments & breakpoints only for Piecewise linear
                _bp_names = [pretty(m['name']) for m in frontier_all[:proj_as_of_idx + 1]]
                if proj_basis == "Piecewise linear":
                    _seg_options = [1, 2, 3] if len(_bp_names) >= 5 else [1, 2]
                    if piecewise_n_segments not in _seg_options:
                        piecewise_n_segments = _seg_options[-1]
                    piecewise_n_segments = st.radio(
                        "Segments", _seg_options,
                        horizontal=True, key="piecewise_n_seg")
                else:
                    # Plain Linear: force 1 segment, clear stale session state
                    piecewise_n_segments = 1
                    st.session_state.pop("piecewise_n_seg", None)
                if piecewise_n_segments >= 2:
                    _default_bp1 = pretty(frontier_all[gpt4o_idx]['name']) if gpt4o_idx <= proj_as_of_idx else _bp_names[len(_bp_names) // 2]
                    _bp1_idx = _bp_names.index(_default_bp1) if _default_bp1 in _bp_names else len(_bp_names) // 2
                    bp1_name = st.selectbox(
                        "Breakpoint", _bp_names[1:],
                        index=max(0, _bp1_idx - 1), key="bp1_select")
                    piecewise_breakpoints.append(bp1_name)
                if piecewise_n_segments >= 3:
                    _bp1_pos = _bp_names.index(bp1_name)
                    _remaining = _bp_names[_bp1_pos + 1:]
                    bp2_name = st.selectbox(
                        "Breakpoint 2", _remaining[:-1],
                        index=len(_remaining[:-1]) // 2, key="bp2_select")
                    piecewise_breakpoints.append(bp2_name)

                # Compute DT defaults from the actual last segment
                if piecewise_n_segments >= 2 and piecewise_breakpoints:
                    _last_bp_idx = _bp_names.index(piecewise_breakpoints[-1]) if piecewise_breakpoints[-1] in _bp_names else 0
                    _pw_seg_days = _pre_days[_last_bp_idx:]
                    _pw_seg_vals = _pre_vals[_last_bp_idx:]
                    if len(_pw_seg_days) >= 2:
                        _pw_seg_params = fit_line(_pw_seg_days, _pw_seg_vals)
                        _pw_seg_dt = round(1.0 / _pw_seg_params[1]) if _pw_seg_params[1] > 0 else _pre_ols_dt
                    else:
                        _pw_seg_dt = _pre_ols_dt
                    _default_dt_lo = max(10, int(round(_pw_seg_dt / 2)))
                    _default_dt_hi = int(round(_pw_seg_dt * 2))
                else:
                    _default_dt_lo = max(10, int(round(_pre_ols_dt / 2)))
                    _default_dt_hi = int(round(_pre_ols_dt * 2))

                # Auto-update DT CIs when segment config changes
                _seg_config = (piecewise_n_segments, tuple(piecewise_breakpoints))
                if st.session_state.get("_metr_seg_config") != _seg_config:
                    st.session_state["_metr_seg_config"] = _seg_config
                    st.session_state.pop("custom_dt_lo", None)
                    st.session_state.pop("custom_dt_hi", None)

                custom_dt_lo, custom_dt_hi = st.columns(2)
                custom_dt_lo = _ss_number_input(custom_dt_lo,
                    "DT CI low (days)", "custom_dt_lo", _default_dt_lo,
                    min_value=10, max_value=2000, step=5)
                custom_dt_hi = _ss_number_input(custom_dt_hi,
                    "DT CI high (days)", "custom_dt_hi", _default_dt_hi,
                    min_value=10, max_value=2000, step=5)
                if custom_dt_lo > custom_dt_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()

                _cur = frontier_all[proj_as_of_idx]
                _def_lo_hrs = (_cur.get(_sb_lo_key) or _cur[_sb_val_key]) / 60
                _def_hi_hrs = (_cur.get(_sb_hi_key) or _cur[_sb_val_key]) / 60
                _pos_lo_col, _pos_hi_col = st.columns(2)
                _p_suffix = "_p80" if _sidebar_p80 else "_p50"
                custom_pos_lo = _ss_number_input(_pos_lo_col,
                    "Pos CI low (h)", "custom_pos_lo" + _p_suffix, round(_def_lo_hrs, 1),
                    min_value=0.01, step=0.5)
                custom_pos_hi = _ss_number_input(_pos_hi_col,
                    "Pos CI high (h)", "custom_pos_hi" + _p_suffix, round(_def_hi_hrs, 1),
                    min_value=0.01, step=0.5)

                custom_dt_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"],
                    horizontal=True, key="custom_dt_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space. "
                         "Log-log: fat right tail.")
                custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal", "Log-log"],
                    horizontal=True, key="custom_pos_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space. "
                         "Log-log: fat right tail.")

        # --- Superexponential controls ---
        superexp_dt_initial = superexp_halflife = None
        superexp_dt_ci_lo = superexp_dt_ci_hi = None
        superexp_pos_lo = superexp_pos_hi = None
        superexp_dt_floor = 30
        is_superexp = False
        if proj_basis == "Superexponential":
            is_superexp = True
            _default_dt_init = 150
            if gpt4o_idx <= proj_as_of_idx:
                _sb_base = frontier_all[0]['date']
                _sb_fr = frontier_all[gpt4o_idx:proj_as_of_idx + 1]
                _sb_days = np.array([(m['date'] - _sb_base).days for m in _sb_fr], dtype=float)
                _sb_log2 = np.array([np.log2(m['p50_min']) for m in _sb_fr])
                _sb_params = fit_line(_sb_days, _sb_log2)
                if _sb_params[1] > 0:
                    _default_dt_init = int(round(1.0 / _sb_params[1]))

            # Pre-compute superexp fit at default halflife to get implied DT for CI defaults
            _pre_se_halflife = 365
            _pre_se_z = 2 ** (_pre_days / _pre_se_halflife)
            _pre_se_X = np.column_stack([np.ones_like(_pre_se_z), _pre_se_z])
            (_pre_se_A, _pre_se_K), *_ = np.linalg.lstsq(_pre_se_X, _pre_vals, rcond=None)
            _pre_se_d_last = _pre_days[-1]
            if _pre_se_K > 0:
                _pre_se_dt = _pre_se_halflife / (_pre_se_K * np.log(2) * 2 ** (_pre_se_d_last / _pre_se_halflife))
            else:
                _pre_se_dt = _pre_ols_dt
            _default_se_dt_lo = max(10, int(round(_pre_se_dt / 2)))
            _default_se_dt_hi = int(round(_pre_se_dt * 2))

            with st.expander("Advanced options"):
                st.button("Reset to defaults", key="reset_superexp",
                          on_click=lambda: st.session_state.update(_reset_metr=True))
                _se_col1, _se_col2 = st.columns(2)
                superexp_dt_initial = _ss_number_input(_se_col1,
                    "Initial DT (days)", "superexp_dt_init", _default_dt_init,
                    min_value=10, max_value=2000, step=5)
                superexp_halflife = _ss_number_input(_se_col2,
                    "DT half-life (days)", "superexp_halflife", 365,
                    min_value=30, max_value=5000, step=30,
                    help="How quickly DT shrinks. Lower = faster.")
                superexp_dt_floor = _ss_number_input(st,
                    "Min DT floor (days)", "superexp_dt_floor", 30,
                    min_value=1, max_value=500, step=5,
                    help="DT can't shrink below this. Prevents runaway projections.")
                _se_ci1, _se_ci2 = st.columns(2)
                superexp_dt_ci_lo = _ss_number_input(_se_ci1,
                    "DT CI low (days)", "superexp_dt_ci_lo", _default_se_dt_lo,
                    min_value=10, max_value=2000, step=5)
                superexp_dt_ci_hi = _ss_number_input(_se_ci2,
                    "DT CI high (days)", "superexp_dt_ci_hi", _default_se_dt_hi,
                    min_value=10, max_value=2000, step=5)
                if superexp_dt_ci_lo > superexp_dt_ci_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()
                _cur = frontier_all[proj_as_of_idx]
                _def_lo_hrs = (_cur.get(_sb_lo_key) or _cur[_sb_val_key]) / 60
                _def_hi_hrs = (_cur.get(_sb_hi_key) or _cur[_sb_val_key]) / 60
                _se_pos1, _se_pos2 = st.columns(2)
                _p_suffix_se = "_p80" if _sidebar_p80 else "_p50"
                superexp_pos_lo = _ss_number_input(_se_pos1,
                    "Pos CI low (h)", "superexp_pos_lo" + _p_suffix_se, round(_def_lo_hrs, 1),
                    min_value=0.01, step=0.5)
                superexp_pos_hi = _ss_number_input(_se_pos2,
                    "Pos CI high (h)", "superexp_pos_hi" + _p_suffix_se, round(_def_hi_hrs, 1),
                    min_value=0.01, step=0.5)

        st.markdown("---")
        show_milestones = st.toggle("Milestones", key="milestones")
        show_labels = st.toggle("Labels", key="labels")
        only_post_gpt4o = st.toggle("GPT-4o+ only", key="post_gpt4o")
        use_p80 = st.toggle("Use p80", key="p80")
        use_log_scale = st.toggle("Log scale", key="log_scale")
        hours_only = st.toggle("Hours only", key="hours_only")

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                frontier_names,
                index=frontier_names.index(proj_as_of_name),
                key='_proj_as_of',
                help="Backtest: project from an earlier model's vantage point.",
            )
            _metr_end_year = st.radio(
                "Project through", [2026, 2027, 2028, 2029],
                horizontal=True, key="metr_end_year")

    # ── Reliability metric keys ──────────────────────────────────────────────
    _val_key = 'p80_min' if use_p80 else 'p50_min'
    _lo_key = 'p80_lo' if use_p80 else 'p50_lo'
    _hi_key = 'p80_hi' if use_p80 else 'p50_hi'
    _reliability_label = "p80" if use_p80 else "p50"

    # ── Build data arrays ────────────────────────────────────────────────────
    frontier_used = frontier_all[:proj_as_of_idx + 1]

    if only_post_gpt4o:
        frontier_plot = list(frontier_all[gpt4o_idx:])
        plot_start_idx = gpt4o_idx
    else:
        frontier_plot = list(frontier_all)
        plot_start_idx = 0

    base_date = frontier_all[0]['date']
    days_all = np.array([(m['date'] - base_date).days for m in frontier_all], dtype=float)
    log2_all = np.array([np.log2(m['p50_min']) for m in frontier_all])

    _fit_start = gpt4o_idx if only_post_gpt4o else 0
    _fit_end = proj_as_of_idx + 1
    frontier_used = frontier_all[_fit_start:_fit_end]
    days_used = days_all[_fit_start:_fit_end]
    log2_used = log2_all[_fit_start:_fit_end]
    log2_disp_used = np.array([np.log2(m[_val_key]) for m in frontier_used])
    n_used = len(frontier_used)

    if proj_basis in ("Linear", "Piecewise linear"):
        # Determine which segment to use for fan starting position
        if piecewise_n_segments >= 2:
            # Build segment indices and fit OLS to last segment
            _bp_names_used = [pretty(m['name']) for m in frontier_used]
            _seg_break_idxs = []
            for bp_name in piecewise_breakpoints:
                if bp_name in _bp_names_used:
                    _seg_break_idxs.append(_bp_names_used.index(bp_name))
            # Last segment: from last breakpoint to end
            _last_seg_start = _seg_break_idxs[-1] if _seg_break_idxs else 0
            _last_seg_range = list(range(_last_seg_start, n_used))
            _cu_params = fit_line(days_used[_last_seg_range], log2_used[_last_seg_range])
        else:
            # Single OLS through all used frontier
            _cu_params = fit_line(days_used, log2_used)
        _cu_current_day = (frontier_used[-1]['date'] - frontier_all[0]['date']).days
        # Intercept: use p50 slope but fit intercept to display (p50 or p80) data
        _cu_log2_disp = np.array([np.log2(m[_val_key]) for m in frontier_used])
        if piecewise_n_segments >= 2:
            _seg_d = days_used[_last_seg_range]
            _seg_y = _cu_log2_disp[_last_seg_range]
        else:
            _seg_d = days_used
            _seg_y = _cu_log2_disp
        _cu_intercept = np.mean(_seg_y - _cu_params[1] * _seg_d)  # best intercept given fixed slope
        _cu_fitted_pos = _cu_intercept + _cu_params[1] * _cu_current_day  # log2(minutes)
        _cu_fitted_hrs = 2**_cu_fitted_pos / 60
        _eff_dt_lo = custom_dt_lo
        _eff_dt_hi = custom_dt_hi
        n_custom = N_SAMPLES
        # Trend: sample doubling times from chosen distribution, centered on OLS slope
        if custom_dt_dist == "Log-log":
            proj_dt = _log_lognormal_from_ci(_eff_dt_lo, _eff_dt_hi, n_custom)
        elif custom_dt_dist == "Lognormal":
            proj_dt = _lognormal_from_ci(_eff_dt_lo, _eff_dt_hi, n_custom)
        else:
            proj_dt = _normal_from_ci(_eff_dt_lo, _eff_dt_hi, n_custom)
        # Position: noise centered on OLS-fitted position, spread from user CI
        if custom_pos_dist == "Log-log":
            _cu_fitted_min = _cu_fitted_hrs * 60
            _cu_pos_sigma_y = (np.log(np.log(custom_pos_hi * 60)) - np.log(np.log(custom_pos_lo * 60))) / (2 * 1.282)
            _cu_pos_mu_y = np.log(np.log(_cu_fitted_min))
            log_min = np.random.lognormal(_cu_pos_mu_y, max(_cu_pos_sigma_y, 0), n_custom)
            proj_start = np.log2(np.exp(log_min))
        elif custom_pos_dist == "Lognormal":
            _cu_pos_sigma = (np.log(custom_pos_hi) - np.log(custom_pos_lo)) / (2 * 1.282)
            _cu_pos_mu = np.log(_cu_fitted_hrs)
            proj_start = np.log2(np.random.lognormal(_cu_pos_mu, max(_cu_pos_sigma, 0), n_custom) * 60)
        else:
            _cu_pos_sigma = (custom_pos_hi - custom_pos_lo) / (2 * 1.282)
            pos_hrs = np.maximum(np.random.normal(_cu_fitted_hrs, max(_cu_pos_sigma, 0), n_custom), custom_pos_lo / 10)
            proj_start = np.log2(pos_hrs * 60)
    elif proj_basis == "Superexponential":
        # Fit y = A + K * 2^(d/halflife) to get trend-consistent starting position
        _se_days = np.array([(m['date'] - frontier_all[0]['date']).days for m in frontier_used], dtype=float)
        _se_log2 = np.array([np.log2(m['p50_min']) for m in frontier_used])
        _se_z = 2 ** (_se_days / superexp_halflife)
        _se_X = np.column_stack([np.ones_like(_se_z), _se_z])
        (_se_A, _se_K), *_ = np.linalg.lstsq(_se_X, _se_log2, rcond=None)
        # Re-fit intercept to display data (p50 or p80) with the same K from p50
        _se_log2_disp = np.array([np.log2(m[_val_key]) for m in frontier_used])
        _se_A_disp = np.mean(_se_log2_disp - _se_K * _se_z)  # best A given fixed K
        # Fitted position at the current model's date (in log2 minutes)
        _se_current_day = (frontier_used[-1]['date'] - frontier_all[0]['date']).days
        _se_fitted_pos = _se_A_disp + _se_K * 2 ** (_se_current_day / superexp_halflife)
        # Implied DT at the current model date from the fit
        # DT(d) = halflife / (K * ln(2) * 2^(d/halflife))
        if _se_K > 0:
            superexp_dt_fitted = superexp_halflife / (_se_K * np.log(2) * 2 ** (_se_current_day / superexp_halflife))
        else:
            superexp_dt_fitted = float('inf')
        n_superexp = N_SAMPLES
        proj_dt = _lognormal_from_ci(superexp_dt_ci_lo, superexp_dt_ci_hi, n_superexp)
        # Position: lognormal noise centered on fitted trend position
        _se_fitted_hrs = 2**_se_fitted_pos / 60
        _se_pos_sigma = (np.log(superexp_pos_hi) - np.log(superexp_pos_lo)) / (2 * 1.282)
        _se_pos_mu = np.log(_se_fitted_hrs)
        proj_start = np.log2(np.random.lognormal(_se_pos_mu, max(_se_pos_sigma, 0), n_superexp) * 60)

    # ── Current SOTA (selected "as of" model) ────────────────────────────────

    current = frontier_used[-1]
    current_log2 = np.log2(current[_val_key])
    current_hrs = current[_val_key] / 60

    # ── Plotly chart ─────────────────────────────────────────────────────────

    proj_end_date = datetime(_metr_end_year, 12, 31)
    proj_n_days = (proj_end_date - current['date']).days + 1
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates = [current['date'] + timedelta(days=int(d)) for d in proj_days_arr]

    # Build all trajectories with correlated (dt, start) pairs
    n_samples = len(proj_dt)
    if is_superexp:
        all_trajectories = proj_start[:, None] + superexp_trajectory(
            proj_days_arr, proj_dt, superexp_halflife, superexp_dt_floor)
    else:
        all_trajectories = proj_start[:, None] + proj_days_arr[None, :] / proj_dt[:, None]

    # y-axis conversion: log2(minutes) -> display value
    def _yconv(log2min):
        """Convert log2(minutes) array/scalar to y-axis value."""
        if use_log_scale:
            return log2min
        return 2**log2min / 60  # hours

    pct5 = _yconv(np.percentile(all_trajectories, 5, axis=0))
    pct10 = _yconv(np.percentile(all_trajectories, 10, axis=0))
    pct25 = _yconv(np.percentile(all_trajectories, 25, axis=0))
    pct50 = _yconv(np.percentile(all_trajectories, 50, axis=0))
    pct75 = _yconv(np.percentile(all_trajectories, 75, axis=0))
    pct90 = _yconv(np.percentile(all_trajectories, 90, axis=0))
    pct95 = _yconv(np.percentile(all_trajectories, 95, axis=0))

    fig = go.Figure()

    # --- Fan bands (toself polygons) ---
    bands_spec = [
        (pct5, pct95, 'rgba(52,152,219,0.10)', '90% CI'),
        (pct10, pct90, 'rgba(52,152,219,0.18)', '80% CI'),
        (pct25, pct75, 'rgba(52,152,219,0.28)', '50% CI'),
    ]
    for lo, hi, color, label in bands_spec:
        x_poly = proj_dates + proj_dates[::-1]
        y_poly = list(hi) + list(lo[::-1])
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly,
            fill='toself', fillcolor=color,
            line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True,
        ))

    # --- Trend lines (the central line of the chart) ---
    # Helper to build hover text for an OLS line over a date range
    def _trend_hover(params, d_start, d_end, base_dt):
        """Build hover texts for an OLS trend line sampled daily. Returns y in display coords."""
        days_range = np.arange(d_start, d_end + 1, 1)
        dates = [base_dt + timedelta(days=int(d)) for d in days_range]
        y_log2 = params[0] + params[1] * days_range
        y_display = _yconv(y_log2)
        texts = []
        for d, y in zip(dates, y_log2):
            h = 2**y / 60
            texts.append(f"{d.strftime('%b %d, %Y')}<br>Trend: {fmt_hrs(h, hours_only=hours_only)}")
        return dates, (y_display.tolist() if hasattr(y_display, 'tolist') else list(y_display)), texts

    if proj_basis in ("Linear", "Piecewise linear"):
        _seg_colors = ['#e74c3c', '#f39c12', '#27ae60']
        if piecewise_n_segments >= 2:
            # Build segment ranges from breakpoint names
            _bp_names_used = [pretty(m['name']) for m in frontier_used]
            _break_idxs = []
            for bp_name in piecewise_breakpoints:
                if bp_name in _bp_names_used:
                    _break_idxs.append(_bp_names_used.index(bp_name))
            # Build segment index ranges (breakpoint included in both adjacent segments)
            _seg_bounds = [0] + _break_idxs + [n_used]
            _segments = []
            for si in range(len(_seg_bounds) - 1):
                end = _seg_bounds[si + 1] + 1 if si < len(_seg_bounds) - 2 else _seg_bounds[si + 1]
                _segments.append(list(range(_seg_bounds[si], min(end, n_used))))
            # Draw each segment
            for si, seg_idx in enumerate(_segments):
                if len(seg_idx) < 2:
                    continue
                seg_params = _fit_slope_p50_intercept_display(days_used[seg_idx], log2_used[seg_idx], log2_disp_used[seg_idx])
                seg_dt = 1.0 / seg_params[1] if seg_params[1] > 0 else float('inf')
                is_last = (si == len(_segments) - 1)
                if is_last:
                    # Historical portion: OLS through data points
                    d0 = int(days_used[seg_idx[0]])
                    d_last = int(days_used[seg_idx[-1]])
                    dates_seg, y_seg, hover_seg = _trend_hover(seg_params, d0, d_last, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_seg,
                        mode='lines', line=dict(color='#2c3e50', width=2.5),
                        name=f'Segment {si+1} ({seg_dt:.0f}d doubling)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
                    # Projected portion: user DT slope from last data point
                    _user_dt_center = np.sqrt(custom_dt_lo * custom_dt_hi)
                    _user_slope = 1.0 / _user_dt_center
                    _ols_val_at_last = seg_params[0] + seg_params[1] * d_last
                    _proj_intercept = _ols_val_at_last - _user_slope * d_last
                    _proj_params = np.array([_proj_intercept, _user_slope])
                    d1 = (proj_end_date - base_date).days
                    dates_proj, y_proj, hover_proj = _trend_hover(_proj_params, d_last, d1, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_proj, y=y_proj,
                        mode='lines', line=dict(color='#2980b9', width=2.5),
                        name=f'Projection ({_user_dt_center:.0f}d doubling, CI {custom_dt_lo}\u2013{custom_dt_hi}d)',
                        hovertext=hover_proj, hoverinfo='text',
                    ))
                else:
                    d0 = int(days_used[seg_idx[0]])
                    d1 = int(days_used[seg_idx[-1]])
                    dates_seg, y_seg, hover_seg = _trend_hover(seg_params, d0, d1, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_seg,
                        mode='lines', line=dict(color=_seg_colors[si % len(_seg_colors)], width=2, dash='dash'),
                        name=f'Segment {si+1} ({seg_dt:.0f}d doubling)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
        else:
            # Single OLS through full used frontier
            custom_params = _fit_slope_p50_intercept_display(days_used, log2_used, log2_disp_used)
            custom_ols_dt = 1.0 / custom_params[1] if custom_params[1] > 0 else float('inf')
            # Historical portion: OLS through data points
            d0 = int(days_used[0])
            d_last = int(days_used[-1])
            dates_seg, y_seg, hover_seg = _trend_hover(custom_params, d0, d_last, base_date)
            fig.add_trace(go.Scatter(
                x=dates_seg, y=y_seg,
                mode='lines', line=dict(color='#2c3e50', width=2.5),
                name=f'OLS trend ({custom_ols_dt:.0f}d doubling)',
                hovertext=hover_seg, hoverinfo='text',
            ))
            # Projected portion: user DT slope from last data point
            _user_dt_center = np.sqrt(custom_dt_lo * custom_dt_hi)
            _user_slope = 1.0 / _user_dt_center
            _ols_val_at_last = custom_params[0] + custom_params[1] * d_last
            _proj_intercept = _ols_val_at_last - _user_slope * d_last
            _proj_params = np.array([_proj_intercept, _user_slope])
            d1 = (proj_end_date - base_date).days
            dates_proj, y_proj, hover_proj = _trend_hover(_proj_params, d_last, d1, base_date)
            fig.add_trace(go.Scatter(
                x=dates_proj, y=y_proj,
                mode='lines', line=dict(color='#2980b9', width=2.5),
                name=f'Projection ({_user_dt_center:.0f}d doubling, CI {custom_dt_lo}\u2013{custom_dt_hi}d)',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif proj_basis == "Superexponential":
        # Historical portion: fit curve through data
        d_start = int(days_used[0])
        d_last = int(days_used[-1])
        days_hist = np.arange(d_start, d_last + 1, 1)
        y_hist = _se_A_disp + _se_K * 2 ** (days_hist / superexp_halflife)
        dates_hist = [base_date + timedelta(days=int(d)) for d in days_hist]
        hover_hist = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {fmt_hrs(2**y / 60, hours_only=hours_only)}" for dt, y in zip(dates_hist, y_hist)]
        y_hist_conv = _yconv(y_hist)
        y_hist_conv = y_hist_conv.tolist() if hasattr(y_hist_conv, 'tolist') else list(y_hist_conv)
        fig.add_trace(go.Scatter(
            x=dates_hist, y=y_hist_conv,
            mode='lines', line=dict(color='#2c3e50', width=2.5),
            name=f'Superexp fit (DT\u2248{superexp_dt_fitted:.0f}d, HL={superexp_halflife}d)',
            hovertext=hover_hist, hoverinfo='text',
        ))
        # Projected portion: use same formula as trajectories with center DT
        _se_user_dt = np.sqrt(superexp_dt_ci_lo * superexp_dt_ci_hi)
        d_end = (proj_end_date - base_date).days
        days_proj = np.arange(0, d_end - d_last + 1, 1)
        y_proj_growth = superexp_trajectory(days_proj, _se_user_dt, superexp_halflife, superexp_dt_floor)
        y_proj_log2 = _se_fitted_pos + y_proj_growth
        dates_proj = [current['date'] + timedelta(days=int(d)) for d in days_proj]
        hover_proj = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {fmt_hrs(2**y / 60, hours_only=hours_only)}" for dt, y in zip(dates_proj, y_proj_log2)]
        y_proj_conv = _yconv(y_proj_log2)
        y_proj_conv = y_proj_conv.tolist() if hasattr(y_proj_conv, 'tolist') else list(y_proj_conv)
        fig.add_trace(go.Scatter(
            x=dates_proj, y=y_proj_conv,
            mode='lines', line=dict(color='#2980b9', width=2.5),
            name=f'Projection (DT={_se_user_dt:.0f}d, CI {superexp_dt_ci_lo}\u2013{superexp_dt_ci_hi}d)',
            hovertext=hover_proj, hoverinfo='text',
        ))

    # --- Milestone hlines ---
    if show_milestones:
        x_lo = frontier_plot[0]['date'] - timedelta(days=30)
        x_hi = proj_end_date
        for hrs, label, color in [
            (8, "1 work-day (8h)", '#888888'),
            (40, "1 work-week (40h)", '#666666'),
            (176, "1 work-month (176h)", '#c0392b'),
        ]:
            lv = _yconv(np.log2(hrs * 60))
            fig.add_trace(go.Scatter(
                x=[x_lo, x_hi], y=[lv, lv],
                mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=1.0, xref='paper', y=lv, text=f"  {label}",
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=color))

    today = _add_today_vline(fig)

    # --- Backtesting ---
    is_backtesting = proj_as_of_idx < len(frontier_all) - 1
    backtest_results = []
    _bt_lookup = {}
    if is_backtesting:
        _bt_future = frontier_all[proj_as_of_idx + 1:]
        backtest_results = _backtest_stats(
            _bt_future, all_trajectories, current['date'], proj_end_date,
            lambda m: np.log2(m[_val_key]),
            lambda m: pretty(m['name']),
        )
        _bt_lookup = {r['name']: r for r in backtest_results}

    # --- Non-SOTA models: only show those within 2 log2 units (4x) of frontier max to reduce clutter ---
    _metr_frontier_max_log2 = max(np.log2(m[_val_key]) for m in frontier_all)
    _metr_nf_cutoff = _metr_frontier_max_log2 - 2
    _gpt4o_date = frontier_all[gpt4o_idx]['date']
    for m in metr_all:
        if m['is_sota']:
            continue
        m_log2 = np.log2(m[_val_key])
        if m_log2 < _metr_nf_cutoff:
            continue
        if only_post_gpt4o and m['date'] < _gpt4o_date:
            continue
        lv = _yconv(m_log2)
        hrs = m[_val_key] / 60
        hover = f"{pretty(m['name'])}<br>{m['date'].strftime('%b %d, %Y')}<br>{hrs:.1f}h"
        fig.add_trace(go.Scatter(
            x=[m['date']], y=[lv],
            mode='markers' + ('+text' if show_labels else ''),
            marker=dict(color='#aaaaaa', size=6, symbol='circle-open',
                        line=dict(color='#bbbbbb', width=1)),
            text=[pretty(m['name'])] if show_labels else None,
            textposition='top right',
            textfont=dict(size=8, color='#bbbbbb'),
            hovertext=hover, hoverinfo='text', showlegend=False,
        ))

    # --- Data points: distinguish used vs future ---
    for idx_m, m in enumerate(frontier_plot):
        global_idx = idx_m + plot_start_idx  # index into frontier_all
        is_used = global_idx <= proj_as_of_idx
        is_selected = global_idx == proj_as_of_idx
        lv = _yconv(np.log2(m[_val_key]))
        hrs = m[_val_key] / 60
        hover = f"{pretty(m['name'])}<br>{m['date'].strftime('%b %d, %Y')}<br>{hrs:.1f}h"

        if is_used:
            # Normal styling for models used in fitting
            color = '#e74c3c' if is_selected else '#4F8DFD'
            sym = 'star' if is_selected else 'circle'
            sz = 14 if is_selected else 10
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[lv],
                mode='markers' + ('+text' if show_labels else ''),
                marker=dict(color=color, size=sz, symbol=sym,
                            line=dict(color='white', width=1)),
                text=[pretty(m['name'])] if show_labels else None,
                textposition='top right',
                textfont=dict(size=9, color='#c0392b' if is_selected else '#1a1a2e'),
                hovertext=hover, hoverinfo='text', showlegend=False,
            ))
            if m.get(_lo_key) and m.get(_hi_key):
                fig.add_trace(go.Scatter(
                    x=[m['date'], m['date']],
                    y=[_yconv(np.log2(m[_lo_key])), _yconv(np.log2(m[_hi_key]))],
                    mode='lines', line=dict(color='#4F8DFD', width=4), opacity=0.2,
                    hoverinfo='skip', showlegend=False,
                ))
        else:
            _bt_name = pretty(m['name'])
            if is_backtesting and _bt_name in _bt_lookup:
                r = _bt_lookup[_bt_name]
                _btc = _bt_color_for(r)
                _bt_label = f"{_bt_name} (p{r['percentile']:.0f})"
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[lv],
                    mode='markers+text',
                    marker=dict(color=_btc, size=12, symbol='diamond',
                                line=dict(color='white', width=1)),
                    text=[_bt_label],
                    textposition='top right',
                    textfont=dict(size=9, color=_btc),
                    hovertext=hover + f"<br>Percentile: {r['percentile']:.0f}%",
                    hoverinfo='text', showlegend=False,
                ))
            else:
                # Grey markers for future models (not used in fitting)
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[lv],
                    mode='markers' + ('+text' if show_labels else ''),
                    marker=dict(color='#aaaaaa', size=10, symbol='circle-open',
                                line=dict(color='#777777', width=2)),
                    text=[pretty(m['name'])] if show_labels else None,
                    textposition='top right',
                    textfont=dict(size=9, color='#999999'),
                    hovertext=hover, hoverinfo='text', showlegend=False,
                ))
            if m.get(_lo_key) and m.get(_hi_key):
                fig.add_trace(go.Scatter(
                    x=[m['date'], m['date']],
                    y=[_yconv(np.log2(m[_lo_key])), _yconv(np.log2(m[_hi_key]))],
                    mode='lines', line=dict(color='#999999', width=3), opacity=0.25,
                    hoverinfo='skip', showlegend=False,
                ))

    # --- Backtest overlay ---
    if is_backtesting and backtest_results:
        _add_backtest_traces(fig, backtest_results, current['date'], yconv=_yconv)

    # --- Layout ---
    if use_log_scale:
        y_min = np.log2(frontier_plot[0][_val_key]) - 1
        y_max = max(pct95[-1], np.log2(176 * 60)) + 2
        tick_vals = list(range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1))
        tick_text = [log2min_to_label(v, hours_only=hours_only) for v in tick_vals]
        yaxis_cfg = dict(
            title=f"{_reliability_label} Horizon Length (log scale)",
            tickvals=tick_vals, ticktext=tick_text,
            range=[y_min, y_max],
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            tickfont=dict(color='#1a1a2e'),
            title_font=dict(color='#1a1a2e'),
        )
    else:
        y_min = 0
        y_max = max(pct95[-1], 176) * 1.1
        yaxis_cfg = dict(
            title=f"{_reliability_label} Horizon Length (hours)",
            range=[y_min, y_max],
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            tickfont=dict(color='#1a1a2e'),
            title_font=dict(color='#1a1a2e'),
        )

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=140, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(
            range=[frontier_plot[0]['date'] - timedelta(days=30),
                   proj_end_date + timedelta(days=30)],
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(color='#1a1a2e'),
            zeroline=False,
        ),
        yaxis=yaxis_cfg,
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)',
                    font=dict(color='#1a1a2e')),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # ── Render chart + metrics ──────────────────────────────────────────────
    st.plotly_chart(fig, width="stretch")
    if is_backtesting and backtest_results:
        _backtest_summary(backtest_results)

    # ── Projections ───────────────────────────────────────────────────────────

    start_hrs_samples = 2**proj_start / 60
    med_dt = np.median(proj_dt)
    p10_dt, p90_dt = np.percentile(proj_dt, [10, 90])
    current_label = pretty(current['name'])

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026EOY", datetime(2026, 12, 31)),
        ("2027 Jun EOM", datetime(2027, 6, 30)),
        ("2027EOY", datetime(2027, 12, 31)),
        ("2028EOY", datetime(2028, 12, 31)),
        ("2029EOY", datetime(2029, 12, 31)),
    ]

    def _proj_hrs_at(elapsed_days, start_hrs, dt_arr, superexp=False, hl=None, floor=None):
        if superexp and hl is not None:
            if floor is not None and floor > 0:
                t_cap = np.where(dt_arr > floor, hl * np.log2(dt_arr / floor), 0.0)
                se_phase = np.minimum(elapsed_days, t_cap)
                doublings_se = (hl / (dt_arr * np.log(2))) * (2**(se_phase / hl) - 1)
                doublings_lin = np.maximum(elapsed_days - t_cap, 0) / floor
                doublings = doublings_se + doublings_lin
            else:
                doublings = (hl / (dt_arr * np.log(2))) * (2**(elapsed_days / hl) - 1)
        else:
            doublings = elapsed_days / dt_arr
        return start_hrs * (2 ** doublings)

    # All columns use the projection model for coherent "all things considered" forecasts
    all_targets = [
        (f"{current_label} ({current['date'].strftime('%b %Y')})", current['date']),
    ] + eoy_targets
    n_all_cols = len(all_targets)
    cols = st.columns([1.2] + [1] * (n_all_cols - 1))
    for col, (label, target_date) in zip(cols, all_targets):
        elapsed = (target_date - current['date']).days
        proj_hrs = _proj_hrs_at(elapsed, start_hrs_samples, proj_dt, is_superexp, superexp_halflife, superexp_dt_floor if is_superexp else None)
        p10_h, p50_h, p90_h = np.percentile(proj_hrs, [10, 50, 90])
        display_h = current_hrs if elapsed == 0 else p50_h
        with col:
            st.metric(label=label, value=fmt_hrs(display_h, hours_only=hours_only))
            st.caption(f"80% CI: {fmt_hrs(p10_h, hours_only=hours_only)} \u2013 {fmt_hrs(p90_h, hours_only=hours_only)}")

    # Milestone tables in expander
    milestone_thresholds = [
        (40, "1 work-week (40h)"),
        (176, "1 work-month (176h)"),
        (2000, "1 work-year (2000h)"),
    ]

    with st.expander("Milestone details"):
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown("**Probabilities**")
            rows = []
            for hrs_threshold, ms_label in milestone_thresholds:
                row = {"Milestone": ms_label}
                for eoy_label, target_date in eoy_targets:
                    elapsed = (target_date - current['date']).days
                    proj_hrs = _proj_hrs_at(elapsed, start_hrs_samples, proj_dt, is_superexp, superexp_halflife, superexp_dt_floor if is_superexp else None)
                    prob = np.mean(proj_hrs >= hrs_threshold) * 100
                    row[eoy_label] = f"{prob:.0f}%"
                rows.append(row)
            st.table(rows)

        with tcol2:
            st.markdown("**Estimated arrival**")
            arrival_rows = []
            for hrs_threshold, ms_label in milestone_thresholds:
                doublings_needed = np.log2(hrs_threshold / start_hrs_samples)
                if is_superexp and superexp_halflife is not None:
                    # Doublings during superexp phase (before DT hits floor)
                    dt_floor = superexp_dt_floor
                    t_cap = np.where(proj_dt > dt_floor, superexp_halflife * np.log2(proj_dt / dt_floor), 0.0)
                    d_at_cap = (superexp_halflife / (proj_dt * np.log(2))) * (2**(t_cap / superexp_halflife) - 1)
                    # If needed doublings fit in superexp phase
                    arg = 1 + doublings_needed * proj_dt * np.log(2) / superexp_halflife
                    arg = np.maximum(arg, 1e-10)
                    days_se_only = superexp_halflife * np.log2(arg)
                    # If they don't, use cap + linear remainder
                    leftover = np.maximum(doublings_needed - d_at_cap, 0)
                    days_with_floor = t_cap + leftover * dt_floor
                    days_to = np.where(doublings_needed <= d_at_cap, days_se_only, days_with_floor)
                else:
                    days_to = doublings_needed * proj_dt
                p10_d, p50_d, p90_d = np.percentile(days_to, [10, 50, 90])
                med_date = current['date'] + timedelta(days=p50_d)
                early_date = current['date'] + timedelta(days=p10_d)
                late_date = current['date'] + timedelta(days=p90_d)
                arrival_rows.append({
                    "Milestone": ms_label,
                    "Median": med_date.strftime('%b %Y'),
                    "80% CI": f"{early_date.strftime('%b %Y')} \u2013 {late_date.strftime('%b %Y')}",
                })
            st.table(arrival_rows)

    _fn_caption(
        "Time units are human work-time: 1d = 8h, 1w = 40h, 1mo = 176h, "
        "1y = 2000h. These are projections, not forecasts.",
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))


# ── Epoch ECI (generic implementation) ──────────────────────────────────

def _eci_tab_reset_keys(p):
    return [
        f"{p}_custom_ppy_lo", f"{p}_custom_ppy_hi",
        f"{p}_custom_pos_lo", f"{p}_custom_pos_hi",
        f"{p}_piecewise_n_seg", f"{p}_bp1_select",
        f"{p}_bp2_select", f"{p}_custom_dpp_dist",
        f"{p}_custom_pos_dist",
        f"{p}_superexp_ppy_init", f"{p}_superexp_halflife",
        f"{p}_superexp_ppy_ceiling", f"{p}_superexp_ppy_ci_lo",
        f"{p}_superexp_ppy_ci_hi", f"{p}_superexp_pos_lo",
        f"{p}_superexp_pos_hi",
        f"{p}_proj_basis", f"{p}_milestones", f"{p}_labels",
        f"{p}_eci_metr_proj",
        f"_{p}_proj_as_of", f"{p}_end_year",
        f"_{p}_seg_config",
    ]

def _eci_tab_defaults(p):
    return {
        f"{p}_proj_basis": "Linear",
        f"{p}_piecewise_n_seg": 1,
        f"{p}_custom_dpp_dist": "Lognormal",
        f"{p}_custom_pos_dist": "Normal",
        f"{p}_milestones": True,
        f"{p}_labels": True,
        f"{p}_eci_metr_proj": False,
        f"{p}_end_year": 2026,
    }

def _eci_metr_hover(eci_score, organization):
    """Return hover text lines with METR horizon projections from ECI score."""
    a = 1 if organization == 'Anthropic' else 0
    # p50 model
    p50_min = 2 ** (0.24 * eci_score + 0.76 * a - 28.68)
    p50_lo, p50_hi = p50_min * 0.66, p50_min * 1.34
    # p80 model
    p80_min = 2 ** (0.23 * eci_score + 0.35 * a - 29.95)
    p80_lo, p80_hi = p80_min * 0.52, p80_min * 1.48
    return (f"<br>METR p50: {fmt_hrs(p50_lo / 60)}\u2013{fmt_hrs(p50_hi / 60)}"
            f"<br>METR p80: {fmt_hrs(p80_lo / 60)}\u2013{fmt_hrs(p80_hi / 60)}")


def _render_eci_tab(tab_all, tab_frontier_all, tab_frontier_names, p,
                    sidebar_header, milestone_list,
                    overlay_frontier=None, overlay_label=None,
                    extra_table_milestones=None, us_best_marker=None,
                    us_match_marker=None, us_match_label=None,
                    overlay_name="US", subject_name=None):
    if st.session_state.pop(f"_reset_{p}", False):
        for k in _eci_tab_reset_keys(p):
            st.session_state.pop(k, None)
        st.session_state.update(_eci_tab_defaults(p))
        st.rerun()

    for k, v in _eci_tab_defaults(p).items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── ECI Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.header(sidebar_header)

        # Read "project as of" from session state
        eci_proj_as_of_name = st.session_state.get(f"_{p}_proj_as_of", tab_frontier_names[-1])
        if eci_proj_as_of_name not in tab_frontier_names:
            eci_proj_as_of_name = tab_frontier_names[-1]
        eci_proj_as_of_idx = tab_frontier_names.index(eci_proj_as_of_name)

        # --- Projection basis ---
        eci_basis_options = ["Linear", "Piecewise linear", "Superexponential"]
        eci_proj_basis = st.radio("Projection basis", eci_basis_options, key=f"{p}_proj_basis")

        eci_custom_dpp_lo = eci_custom_dpp_hi = None
        eci_custom_pos_lo = eci_custom_pos_hi = None
        eci_custom_dpp_dist = "Lognormal"
        eci_custom_pos_dist = "Lognormal"
        eci_piecewise_n_segments = 1
        eci_piecewise_breakpoints = []
        _eci_is_linear = eci_proj_basis in ("Linear", "Piecewise linear")
        if eci_proj_basis == "Piecewise linear":
            eci_piecewise_n_segments = 2

        # Pre-compute OLS PPY for data-driven defaults
        _eci_pre_fr = [m for m in tab_all if m['is_frontier']][:eci_proj_as_of_idx + 1]
        _eci_pre_base = _eci_pre_fr[0]['date'] if _eci_pre_fr else tab_frontier_all[0]['date']
        _eci_pre_days = np.array([(m['date'] - _eci_pre_base).days for m in _eci_pre_fr], dtype=float)
        _eci_pre_scores = np.array([m['eci_score'] for m in _eci_pre_fr])
        _eci_pre_params = fit_line(_eci_pre_days, _eci_pre_scores) if len(_eci_pre_fr) >= 2 else np.array([0, 0.046])
        _eci_pre_ppy = round(_eci_pre_params[1] * 365.25, 1) if _eci_pre_params[1] > 0 else 16.9

        if _eci_is_linear:
            with st.expander("Advanced options"):
                st.button("Reset to defaults", key=f"reset_{p}_linear",
                          on_click=lambda: st.session_state.update({f"_reset_{p}": True}))

                # Segments & breakpoints only for Piecewise linear
                _eci_bp_names = [m['display_name'] for m in tab_frontier_all[:eci_proj_as_of_idx + 1]]
                if eci_proj_basis == "Piecewise linear":
                    _eci_seg_options = [1, 2, 3] if len(_eci_bp_names) >= 5 else [1, 2]
                    if eci_piecewise_n_segments not in _eci_seg_options:
                        eci_piecewise_n_segments = _eci_seg_options[-1]
                    # Ensure session state defaults to 2 for Piecewise
                    if st.session_state.get(f"{p}_piecewise_n_seg", 1) < 2:
                        st.session_state[f"{p}_piecewise_n_seg"] = 2
                    eci_piecewise_n_segments = st.radio(
                        "Segments", _eci_seg_options,
                        horizontal=True, key=f"{p}_piecewise_n_seg")
                else:
                    # Plain Linear: force 1 segment, clear stale session state
                    eci_piecewise_n_segments = 1
                    st.session_state.pop(f"{p}_piecewise_n_seg", None)
                if eci_piecewise_n_segments >= 2:
                    _eci_default_bp1 = _eci_bp_names[len(_eci_bp_names) // 2]
                    _eci_bp1_idx = _eci_bp_names.index(_eci_default_bp1) if _eci_default_bp1 in _eci_bp_names else len(_eci_bp_names) // 2
                    eci_bp1_name = st.selectbox(
                        "Breakpoint", _eci_bp_names[1:],
                        index=max(0, _eci_bp1_idx - 1), key=f"{p}_bp1_select")
                    eci_piecewise_breakpoints.append(eci_bp1_name)
                if eci_piecewise_n_segments >= 3:
                    _eci_bp1_pos = _eci_bp_names.index(eci_bp1_name)
                    _eci_remaining = _eci_bp_names[_eci_bp1_pos + 1:]
                    eci_bp2_name = st.selectbox(
                        "Breakpoint 2", _eci_remaining[:-1],
                        index=len(_eci_remaining[:-1]) // 2, key=f"{p}_bp2_select")
                    eci_piecewise_breakpoints.append(eci_bp2_name)

                # Compute PPY defaults from the actual last segment
                if eci_piecewise_n_segments >= 2 and eci_piecewise_breakpoints:
                    _eci_last_bp_idx = _eci_bp_names.index(eci_piecewise_breakpoints[-1]) if eci_piecewise_breakpoints[-1] in _eci_bp_names else 0
                    _eci_pw_seg_days = _eci_pre_days[_eci_last_bp_idx:]
                    _eci_pw_seg_scores = _eci_pre_scores[_eci_last_bp_idx:]
                    if len(_eci_pw_seg_days) >= 2:
                        _eci_pw_seg_params = fit_line(_eci_pw_seg_days, _eci_pw_seg_scores)
                        _eci_pw_seg_ppy = round(_eci_pw_seg_params[1] * 365.25, 1) if _eci_pw_seg_params[1] > 0 else _eci_pre_ppy
                    else:
                        _eci_pw_seg_ppy = _eci_pre_ppy
                    _eci_default_ppy_lo = round(_eci_pw_seg_ppy / 2, 1)
                    _eci_default_ppy_hi = round(_eci_pw_seg_ppy * 2, 1)
                else:
                    _eci_default_ppy_lo = round(_eci_pre_ppy / 2, 1)
                    _eci_default_ppy_hi = round(_eci_pre_ppy * 2, 1)

                # Auto-update PPY CIs when segment config changes
                _eci_seg_config = (eci_piecewise_n_segments, tuple(eci_piecewise_breakpoints))
                if st.session_state.get(f"_{p}_seg_config") != _eci_seg_config:
                    st.session_state[f"_{p}_seg_config"] = _eci_seg_config
                    st.session_state.pop(f"{p}_custom_ppy_lo", None)
                    st.session_state.pop(f"{p}_custom_ppy_hi", None)

                _eci_ppy_lo_col, _eci_ppy_hi_col = st.columns(2)
                eci_custom_ppy_lo = _ss_number_input(_eci_ppy_lo_col,
                    "+Pts/Yr CI low", f"{p}_custom_ppy_lo", _eci_default_ppy_lo,
                    min_value=0.5, max_value=365.0, step=0.5)
                eci_custom_ppy_hi = _ss_number_input(_eci_ppy_hi_col,
                    "+Pts/Yr CI high", f"{p}_custom_ppy_hi", _eci_default_ppy_hi,
                    min_value=0.5, max_value=365.0, step=0.5)
                if eci_custom_ppy_lo > eci_custom_ppy_hi:
                    st.error("+Pts/Yr CI low must be ≤ +Pts/Yr CI high.")
                    st.stop()
                eci_custom_dpp_lo = 365.25 / eci_custom_ppy_hi  # high PPY = low DPP (fast)
                eci_custom_dpp_hi = 365.25 / eci_custom_ppy_lo  # low PPY = high DPP (slow)

                # Position CI: fitted score +/- 2
                _eci_cur = tab_frontier_all[eci_proj_as_of_idx]
                _eci_def_score = _eci_cur['eci_score']
                _eci_pos_lo_col, _eci_pos_hi_col = st.columns(2)
                eci_custom_pos_lo = _ss_number_input(_eci_pos_lo_col,
                    "Pos CI low (ECI)", f"{p}_custom_pos_lo", round(_eci_def_score - 2, 1),
                    step=0.5)
                eci_custom_pos_hi = _ss_number_input(_eci_pos_hi_col,
                    "Pos CI high (ECI)", f"{p}_custom_pos_hi", round(_eci_def_score + 2, 1),
                    step=0.5)

                eci_custom_dpp_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"],
                    horizontal=True, key=f"{p}_custom_dpp_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space. "
                         "Log-log: fat right tail.")
                eci_custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal"],
                    horizontal=True, key=f"{p}_custom_pos_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space.")

        # --- Superexponential controls ---
        eci_superexp_dpp_initial = eci_superexp_halflife = None
        eci_superexp_dpp_ci_lo = eci_superexp_dpp_ci_hi = None
        eci_superexp_pos_lo = eci_superexp_pos_hi = None
        eci_superexp_dpp_floor = 10
        eci_is_superexp = False
        if eci_proj_basis == "Superexponential":
            eci_is_superexp = True
            _eci_default_ppy_init = 10.0
            # Estimate from recent frontier
            if len(tab_frontier_all[:eci_proj_as_of_idx + 1]) >= 2:
                _eci_base = tab_frontier_all[0]['date']
                _eci_fr = tab_frontier_all[:eci_proj_as_of_idx + 1]
                _eci_fd = np.array([(m['date'] - _eci_base).days for m in _eci_fr], dtype=float)
                _eci_fs = np.array([m['eci_score'] for m in _eci_fr])
                _eci_fp = fit_line(_eci_fd, _eci_fs)
                if _eci_fp[1] > 0:
                    _eci_default_ppy_init = round(365.25 * _eci_fp[1], 1)

            # Pre-compute superexp fit at default halflife to get implied PPY for CI defaults
            _eci_pre_se_halflife = 365
            _eci_pre_se_z = 2 ** (_eci_pre_days / _eci_pre_se_halflife)
            _eci_pre_se_X = np.column_stack([np.ones_like(_eci_pre_se_z), _eci_pre_se_z])
            (_eci_pre_se_A, _eci_pre_se_K), *_ = np.linalg.lstsq(_eci_pre_se_X, _eci_pre_scores, rcond=None)
            _eci_pre_se_d_last = _eci_pre_days[-1]
            if _eci_pre_se_K > 0:
                _eci_pre_se_dpp = _eci_pre_se_halflife / (_eci_pre_se_K * np.log(2) * 2 ** (_eci_pre_se_d_last / _eci_pre_se_halflife))
                _eci_pre_se_ppy = round(365.25 / _eci_pre_se_dpp, 1)
            else:
                _eci_pre_se_ppy = _eci_pre_ppy
            _eci_default_se_ppy_lo = round(max(0.5, _eci_pre_se_ppy / 2), 1)
            _eci_default_se_ppy_hi = round(_eci_pre_se_ppy * 2, 1)

            with st.expander("Advanced options"):
                st.button("Reset to defaults", key=f"reset_{p}_superexp",
                          on_click=lambda: st.session_state.update({f"_reset_{p}": True}))
                _eci_se_col1, _eci_se_col2 = st.columns(2)
                eci_superexp_ppy_initial = _ss_number_input(_eci_se_col1,
                    "Initial +Pts/Yr", f"{p}_superexp_ppy_init", _eci_default_ppy_init,
                    min_value=0.5, max_value=365.0, step=0.5)
                eci_superexp_dpp_initial = 365.25 / eci_superexp_ppy_initial
                eci_superexp_halflife = _ss_number_input(_eci_se_col2,
                    "Rate half-life (days)", f"{p}_superexp_halflife", 365,
                    min_value=30, max_value=5000, step=30,
                    help="How quickly rate grows. Lower = faster.")
                eci_superexp_ppy_ceiling = _ss_number_input(st,
                    "Max +Pts/Yr ceiling", f"{p}_superexp_ppy_ceiling", 37.0,
                    min_value=1.0, max_value=365.0, step=1.0,
                    help="Rate can't exceed this. Prevents runaway projections.")
                eci_superexp_dpp_floor = 365.25 / eci_superexp_ppy_ceiling
                _eci_se_ci1, _eci_se_ci2 = st.columns(2)
                eci_superexp_ppy_ci_lo = _ss_number_input(_eci_se_ci1,
                    "+Pts/Yr CI low", f"{p}_superexp_ppy_ci_lo", _eci_default_se_ppy_lo,
                    min_value=0.5, max_value=365.0, step=0.5)
                eci_superexp_ppy_ci_hi = _ss_number_input(_eci_se_ci2,
                    "+Pts/Yr CI high", f"{p}_superexp_ppy_ci_hi", _eci_default_se_ppy_hi,
                    min_value=0.5, max_value=365.0, step=0.5)
                if eci_superexp_ppy_ci_lo > eci_superexp_ppy_ci_hi:
                    st.error("+Pts/Yr CI low must be ≤ +Pts/Yr CI high.")
                    st.stop()
                eci_superexp_dpp_ci_lo = 365.25 / eci_superexp_ppy_ci_hi  # high PPY = low DPP
                eci_superexp_dpp_ci_hi = 365.25 / eci_superexp_ppy_ci_lo  # low PPY = high DPP
                _eci_cur = tab_frontier_all[eci_proj_as_of_idx]
                _eci_def_score = _eci_cur['eci_score']
                _eci_se_pos1, _eci_se_pos2 = st.columns(2)
                eci_superexp_pos_lo = _ss_number_input(_eci_se_pos1,
                    "Pos CI low (ECI)", f"{p}_superexp_pos_lo", round(_eci_def_score - 2, 1),
                    step=0.5)
                eci_superexp_pos_hi = _ss_number_input(_eci_se_pos2,
                    "Pos CI high (ECI)", f"{p}_superexp_pos_hi", round(_eci_def_score + 2, 1),
                    step=0.5)

        st.markdown("---")
        eci_show_milestones = st.toggle("Milestones", key=f"{p}_milestones")
        eci_show_labels = st.toggle("Labels", key=f"{p}_labels")
        eci_metr_proj = st.toggle("ECI→METR projections", key=f"{p}_eci_metr_proj")

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                tab_frontier_names,
                index=tab_frontier_names.index(eci_proj_as_of_name),
                key=f"_{p}_proj_as_of",
                help="Backtest: project from an earlier model's vantage point.",
            )
            _eci_end_year = st.radio(
                "Project through", [2026, 2027, 2028, 2029],
                horizontal=True, key=f"{p}_end_year")

    # ── Build data arrays ────────────────────────────────────────────────────
    eci_frontier_used = tab_frontier_all[:eci_proj_as_of_idx + 1]
    eci_frontier_plot = list(tab_all)  # show all models (frontier + non-frontier)

    base_date = tab_frontier_all[0]['date']
    days_all_eci = np.array([(m['date'] - base_date).days for m in tab_frontier_all], dtype=float)
    scores_all_eci = np.array([m['eci_score'] for m in tab_frontier_all])

    _eci_fit_start = 0
    _eci_fit_end = eci_proj_as_of_idx + 1
    eci_frontier_used = tab_frontier_all[_eci_fit_start:_eci_fit_end]
    days_used = days_all_eci[_eci_fit_start:_eci_fit_end]
    scores_used = scores_all_eci[_eci_fit_start:_eci_fit_end]
    n_used = len(eci_frontier_used)

    # DPP = days per point (analogous to doubling time but for linear ECI score)
    # score(t) = intercept + slope * t  =>  dpp = 1/slope

    if eci_proj_basis in ("Linear", "Piecewise linear"):
        if eci_piecewise_n_segments >= 2:
            _eci_bp_names_used = [m['display_name'] for m in eci_frontier_used]
            _eci_seg_break_idxs = []
            for bp_name in eci_piecewise_breakpoints:
                if bp_name in _eci_bp_names_used:
                    _eci_seg_break_idxs.append(_eci_bp_names_used.index(bp_name))
            _eci_last_seg_start = _eci_seg_break_idxs[-1] if _eci_seg_break_idxs else 0
            _eci_last_seg_range = list(range(_eci_last_seg_start, n_used))
            _eci_params = fit_line(days_used[_eci_last_seg_range], scores_used[_eci_last_seg_range])
        else:
            _eci_params = fit_line(days_used, scores_used)

        _eci_current_day = (eci_frontier_used[-1]['date'] - base_date).days
        if eci_piecewise_n_segments >= 2:
            _eci_seg_d = days_used[_eci_last_seg_range]
            _eci_seg_y = scores_used[_eci_last_seg_range]
        else:
            _eci_seg_d = days_used
            _eci_seg_y = scores_used
        _eci_intercept = np.mean(_eci_seg_y - _eci_params[1] * _eci_seg_d)
        _eci_fitted_score = _eci_intercept + _eci_params[1] * _eci_current_day

        _eci_eff_dpp_lo = eci_custom_dpp_lo
        _eci_eff_dpp_hi = eci_custom_dpp_hi

        n_eci = N_SAMPLES
        if eci_custom_dpp_dist == "Log-log":
            eci_proj_dpp = _log_lognormal_from_ci(_eci_eff_dpp_lo, _eci_eff_dpp_hi, n_eci)
        elif eci_custom_dpp_dist == "Lognormal":
            eci_proj_dpp = _lognormal_from_ci(_eci_eff_dpp_lo, _eci_eff_dpp_hi, n_eci)
        else:
            eci_proj_dpp = _normal_from_ci(_eci_eff_dpp_lo, _eci_eff_dpp_hi, n_eci)

        # Position samples centered on OLS-fitted position
        if eci_custom_pos_dist == "Lognormal":
            _eci_pos_offset = 50  # shift so values are safely positive
            _eci_pos_sigma = (np.log(eci_custom_pos_hi + _eci_pos_offset) - np.log(eci_custom_pos_lo + _eci_pos_offset)) / (2 * 1.282)
            _eci_pos_mu = np.log(_eci_fitted_score + _eci_pos_offset)
            eci_proj_start = np.random.lognormal(_eci_pos_mu, max(_eci_pos_sigma, 0), n_eci) - _eci_pos_offset
        else:
            _eci_pos_sigma = (eci_custom_pos_hi - eci_custom_pos_lo) / (2 * 1.282)
            eci_proj_start = np.random.normal(_eci_fitted_score, max(_eci_pos_sigma, 0), n_eci)

    elif eci_proj_basis == "Superexponential":
        # Fit score = A + K * 2^(d/halflife)
        _eci_se_days = np.array([(m['date'] - base_date).days for m in eci_frontier_used], dtype=float)
        _eci_se_scores = np.array([m['eci_score'] for m in eci_frontier_used])
        _eci_se_z = 2 ** (_eci_se_days / eci_superexp_halflife)
        _eci_se_X = np.column_stack([np.ones_like(_eci_se_z), _eci_se_z])
        (_eci_se_A, _eci_se_K), *_ = np.linalg.lstsq(_eci_se_X, _eci_se_scores, rcond=None)

        _eci_se_current_day = (eci_frontier_used[-1]['date'] - base_date).days
        _eci_se_fitted_score = _eci_se_A + _eci_se_K * 2 ** (_eci_se_current_day / eci_superexp_halflife)

        # Implied DPP at current date
        if _eci_se_K > 0:
            eci_superexp_dpp_fitted = eci_superexp_halflife / (_eci_se_K * np.log(2) * 2 ** (_eci_se_current_day / eci_superexp_halflife))
        else:
            eci_superexp_dpp_fitted = float('inf')

        n_eci = N_SAMPLES
        eci_proj_dpp = _lognormal_from_ci(eci_superexp_dpp_ci_lo, eci_superexp_dpp_ci_hi, n_eci)

        # Position: normal noise centered on fitted trend position
        _eci_se_pos_sigma = (eci_superexp_pos_hi - eci_superexp_pos_lo) / (2 * 1.282)
        eci_proj_start = np.random.normal(_eci_se_fitted_score, max(_eci_se_pos_sigma, 0), n_eci)

    # ── Current SOTA ──────────────────────────────────────────────────────
    eci_current = eci_frontier_used[-1]
    eci_current_score = eci_current['eci_score']

    # ── Build trajectories ────────────────────────────────────────────────
    proj_end_date = datetime(_eci_end_year, 12, 31)
    proj_n_days = (proj_end_date - eci_current['date']).days + 1
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates = [eci_current['date'] + timedelta(days=int(d)) for d in proj_days_arr]

    n_samples = len(eci_proj_dpp)
    if eci_is_superexp:
        all_trajectories = eci_proj_start[:, None] + superexp_trajectory(
            proj_days_arr, eci_proj_dpp, eci_superexp_halflife, eci_superexp_dpp_floor)
    else:
        all_trajectories = eci_proj_start[:, None] + proj_days_arr[None, :] / eci_proj_dpp[:, None]

    pct5 = np.percentile(all_trajectories, 5, axis=0)
    pct10 = np.percentile(all_trajectories, 10, axis=0)
    pct25 = np.percentile(all_trajectories, 25, axis=0)
    pct50 = np.percentile(all_trajectories, 50, axis=0)
    pct75 = np.percentile(all_trajectories, 75, axis=0)
    pct90 = np.percentile(all_trajectories, 90, axis=0)
    pct95 = np.percentile(all_trajectories, 95, axis=0)

    fig = go.Figure()

    # --- Fan bands ---
    bands_spec = [
        (pct5, pct95, 'rgba(52,152,219,0.10)', '90% CI'),
        (pct10, pct90, 'rgba(52,152,219,0.18)', '80% CI'),
        (pct25, pct75, 'rgba(52,152,219,0.28)', '50% CI'),
    ]
    for lo, hi, color, label in bands_spec:
        x_poly = proj_dates + proj_dates[::-1]
        y_poly = list(hi) + list(lo[::-1])
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly,
            fill='toself', fillcolor=color,
            line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True,
        ))

    # Comparison-trend gap helpers (months the primary entity lags the overlay
    # entity's trend at a given score). Only active when an overlay frontier is
    # supplied. `overlay_name` labels the compared-against entity in hovers.
    _eci_gap_fn = None
    if overlay_frontier is not None and len(overlay_frontier) >= 2:
        # Same gap-in-time metric as the US-China section; share one definition.
        _us_fr_tuples = [(mm['date'], mm['eci_score'], '') for mm in overlay_frontier]

        def _eci_gap_fn(score, at_date):
            """Months `score` lags the overlay trend, evaluated at `at_date`."""
            return _eci_months_behind(_us_fr_tuples, score, at_date)

    def _eci_gap_hover(score, at_date):
        """'<br>Gap: X mo behind <overlay>' suffix, or '' with no overlay frontier."""
        if _eci_gap_fn is None:
            return ""
        g = _eci_gap_fn(score, at_date)
        if g >= 0:
            return f"<br>Gap: {g:.1f} mo behind {overlay_name}"
        return f"<br>Gap: {-g:.1f} mo ahead of {overlay_name}"

    # --- Trend lines ---
    def _eci_trend_hover(params, d_start, d_end, base_dt):
        """Build hover texts for an OLS trend line on ECI scores."""
        days_range = np.arange(d_start, d_end + 1, 1)
        dates = [base_dt + timedelta(days=int(d)) for d in days_range]
        y_scores = params[0] + params[1] * days_range
        texts = []
        for d, y in zip(dates, y_scores):
            texts.append(f"{d.strftime('%b %d, %Y')}<br>Trend: {y:.1f}{_eci_gap_hover(y, d)}")
        return dates, y_scores.tolist(), texts

    if eci_proj_basis in ("Linear", "Piecewise linear"):
        _seg_colors = ['#e74c3c', '#f39c12', '#27ae60']
        if eci_piecewise_n_segments >= 2:
            _eci_bp_names_used = [m['display_name'] for m in eci_frontier_used]
            _eci_break_idxs = []
            for bp_name in eci_piecewise_breakpoints:
                if bp_name in _eci_bp_names_used:
                    _eci_break_idxs.append(_eci_bp_names_used.index(bp_name))
            _eci_seg_bounds = [0] + _eci_break_idxs + [n_used]
            _eci_segments = []
            for si in range(len(_eci_seg_bounds) - 1):
                end = _eci_seg_bounds[si + 1] + 1 if si < len(_eci_seg_bounds) - 2 else _eci_seg_bounds[si + 1]
                _eci_segments.append(list(range(_eci_seg_bounds[si], min(end, n_used))))
            for si, seg_idx in enumerate(_eci_segments):
                if len(seg_idx) < 2:
                    continue
                seg_params = fit_line(days_used[seg_idx], scores_used[seg_idx])
                seg_dpp = 1.0 / seg_params[1] if seg_params[1] > 0 else float('inf')
                is_last = (si == len(_eci_segments) - 1)
                if is_last:
                    # Historical portion: OLS through data points
                    d0 = int(days_used[seg_idx[0]])
                    d_last = int(days_used[seg_idx[-1]])
                    dates_seg, y_seg, hover_seg = _eci_trend_hover(seg_params, d0, d_last, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_seg,
                        mode='lines', line=dict(color='#2c3e50', width=2.5),
                        name=f'Segment {si+1} ({365.25/seg_dpp:.1f} pts/yr)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
                    # Projected portion: user DPP slope from last data point
                    _user_dpp_center = np.sqrt(eci_custom_dpp_lo * eci_custom_dpp_hi)
                    _user_ppy_center = 365.25 / _user_dpp_center
                    _user_slope = 1.0 / _user_dpp_center
                    _ols_val_at_last = seg_params[0] + seg_params[1] * d_last
                    _proj_intercept = _ols_val_at_last - _user_slope * d_last
                    _proj_params = np.array([_proj_intercept, _user_slope])
                    d1 = (proj_end_date - base_date).days
                    dates_proj, y_proj, hover_proj = _eci_trend_hover(_proj_params, d_last, d1, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_proj, y=y_proj,
                        mode='lines', line=dict(color='#2980b9', width=2.5),
                        name=f'Projection ({_user_ppy_center:.1f} pts/yr, CI {eci_custom_ppy_lo}\u2013{eci_custom_ppy_hi})',
                        hovertext=hover_proj, hoverinfo='text',
                    ))
                else:
                    d0 = int(days_used[seg_idx[0]])
                    d1 = int(days_used[seg_idx[-1]])
                    dates_seg, y_seg, hover_seg = _eci_trend_hover(seg_params, d0, d1, base_date)
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_seg,
                        mode='lines', line=dict(color=_seg_colors[si % len(_seg_colors)], width=2, dash='dash'),
                        name=f'Segment {si+1} ({365.25/seg_dpp:.1f} pts/yr)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
        else:
            eci_ols_params = fit_line(days_used, scores_used)
            eci_ols_dpp = 1.0 / eci_ols_params[1] if eci_ols_params[1] > 0 else float('inf')
            # Historical portion: OLS through data points
            d0 = int(days_used[0])
            d_last = int(days_used[-1])
            dates_seg, y_seg, hover_seg = _eci_trend_hover(eci_ols_params, d0, d_last, base_date)
            fig.add_trace(go.Scatter(
                x=dates_seg, y=y_seg,
                mode='lines', line=dict(color='#2c3e50', width=2.5),
                name=f'OLS trend ({365.25/eci_ols_dpp:.1f} pts/yr)',
                hovertext=hover_seg, hoverinfo='text',
            ))
            # Projected portion: user DPP slope from last data point
            _user_dpp_center = np.sqrt(eci_custom_dpp_lo * eci_custom_dpp_hi)
            _user_ppy_center = 365.25 / _user_dpp_center
            _user_slope = 1.0 / _user_dpp_center
            _ols_val_at_last = eci_ols_params[0] + eci_ols_params[1] * d_last
            _proj_intercept = _ols_val_at_last - _user_slope * d_last
            _proj_params = np.array([_proj_intercept, _user_slope])
            d1 = (proj_end_date - base_date).days
            dates_proj, y_proj, hover_proj = _eci_trend_hover(_proj_params, d_last, d1, base_date)
            fig.add_trace(go.Scatter(
                x=dates_proj, y=y_proj,
                mode='lines', line=dict(color='#2980b9', width=2.5),
                name=f'Projection ({_user_ppy_center:.1f} pts/yr, CI {eci_custom_ppy_lo}\u2013{eci_custom_ppy_hi})',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif eci_proj_basis == "Superexponential":
        # Historical portion: fit curve through data
        d_start = int(days_used[0])
        d_last = int(days_used[-1])
        days_hist = np.arange(d_start, d_last + 1, 1)
        y_hist = _eci_se_A + _eci_se_K * 2 ** (days_hist / eci_superexp_halflife)
        dates_hist = [base_date + timedelta(days=int(d)) for d in days_hist]
        hover_hist = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.1f}{_eci_gap_hover(y, dt)}" for dt, y in zip(dates_hist, y_hist)]
        fig.add_trace(go.Scatter(
            x=dates_hist, y=y_hist.tolist(),
            mode='lines', line=dict(color='#2c3e50', width=2.5),
            name=f'Superexp fit ({365.25/eci_superexp_dpp_fitted:.1f} pts/yr, HL={eci_superexp_halflife}d)',
            hovertext=hover_hist, hoverinfo='text',
        ))
        # Projected portion: use same formula as trajectories with center DPP
        _eci_user_dpp = np.sqrt(eci_superexp_dpp_ci_lo * eci_superexp_dpp_ci_hi)
        d_end = (proj_end_date - base_date).days
        days_proj = np.arange(0, d_end - d_last + 1, 1)
        y_proj_growth = superexp_trajectory(days_proj, _eci_user_dpp, eci_superexp_halflife, eci_superexp_dpp_floor)
        y_proj = _eci_se_fitted_score + y_proj_growth
        dates_proj = [eci_current['date'] + timedelta(days=int(d)) for d in days_proj]
        hover_proj = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.1f}{_eci_gap_hover(y, dt)}" for dt, y in zip(dates_proj, y_proj)]
        fig.add_trace(go.Scatter(
            x=dates_proj, y=y_proj.tolist(),
            mode='lines', line=dict(color='#2980b9', width=2.5),
            name=f'Projection (DPP={_eci_user_dpp:.0f}d, CI {eci_superexp_dpp_ci_lo:.0f}\u2013{eci_superexp_dpp_ci_hi:.0f}d)',
            hovertext=hover_proj, hoverinfo='text',
        ))

    # --- Milestone hlines ---
    if eci_show_milestones:
        x_lo = tab_all[0]['date'] - timedelta(days=30)
        x_hi = proj_end_date

        def _add_us_dot(marker, color, textpos='top center'):
            """Draw a labeled dot at a US model, with a dotted line emanating right."""
            fig.add_trace(go.Scatter(
                x=[marker['date'], x_hi], y=[marker['score'], marker['score']],
                mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=[marker['date']], y=[marker['score']],
                mode='markers+text',
                marker=dict(color=color, size=10, symbol='circle',
                            line=dict(color='white', width=1)),
                text=[marker['name']], textposition=textpos,
                textfont=dict(size=10, color=color),
                hovertext=[f"{marker['name']}<br>"
                           f"{marker['date'].strftime('%b %d, %Y')}<br>"
                           f"ECI {marker['score']:.1f}"],
                hoverinfo='text', showlegend=False,
            ))

        for score_val, label, color in milestone_list:
            # The US-best line emanates rightward from the model's release date
            # rather than spanning the full chart width.
            _is_us_best = (us_best_marker is not None
                           and abs(score_val - us_best_marker['score']) < 1e-6)
            if not _is_us_best:
                fig.add_trace(go.Scatter(
                    x=[x_lo, x_hi], y=[score_val, score_val],
                    mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ))
            fig.add_annotation(
                x=1.0, xref='paper', y=score_val, text=f"  {label}",
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=color))
            if _is_us_best:
                _add_us_dot(us_best_marker, color)

        if us_match_marker is not None:
            _add_us_dot(us_match_marker, '#8e44ad', textpos='bottom center')
            if us_match_label:
                fig.add_annotation(
                    x=1.0, xref='paper', y=us_match_marker['score'],
                    text=f"  {us_match_label}", showarrow=False,
                    xanchor='left', yanchor='middle',
                    font=dict(size=10, color='#8e44ad'))

        # Third comparison marker: the benchmark model nearest where the
        # subject's median projection sits today. Only meaningful when the
        # benchmark actually reached that level, and skipped when it lands on
        # a model already marked (best / current-best match).
        if overlay_frontier is not None and us_best_marker is not None:
            _pt_today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            _pt_idx = (_pt_today - eci_current['date']).days
            if 0 <= _pt_idx < len(pct50):
                _proj_today_score = float(pct50[_pt_idx])
                _ovl_best_score = max(m['eci_score'] for m in overlay_frontier)
                _taken = {m['name'] for m in (us_best_marker, us_match_marker) if m}
                if _ovl_best_score >= _proj_today_score:
                    _proj_match = min(
                        overlay_frontier,
                        key=lambda m: abs(m['eci_score'] - _proj_today_score))
                    if _proj_match['display_name'] not in _taken:
                        _add_us_dot({
                            'date': _proj_match['date'],
                            'score': _proj_match['eci_score'],
                            'name': _proj_match['display_name'],
                        }, '#a569bd', textpos='bottom center')
                        fig.add_annotation(
                            x=1.0, xref='paper', y=_proj_match['eci_score'],
                            text=f"  {subject_name or 'Subject'} proj. today {_proj_today_score:.1f}",
                            showarrow=False, xanchor='left', yanchor='middle',
                            font=dict(size=10, color='#a569bd'))

    # --- Overlay trendline (e.g., US trend on the China chart) ---
    if overlay_frontier is not None and len(overlay_frontier) >= 2:
        _ovl_base = overlay_frontier[0]['date']
        _ovl_days = np.array([(m['date'] - _ovl_base).days for m in overlay_frontier], dtype=float)
        _ovl_scores = np.array([m['eci_score'] for m in overlay_frontier])
        _ovl_params = fit_line(_ovl_days, _ovl_scores)
        _ovl_ppy = _ovl_params[1] * 365.25
        _ovl_x_start = overlay_frontier[0]['date']
        _ovl_x_end = proj_end_date
        _ovl_y_start = _ovl_params[0]
        _ovl_y_end = _ovl_params[0] + _ovl_params[1] * (_ovl_x_end - _ovl_base).days
        fig.add_trace(go.Scatter(
            x=[_ovl_x_start, _ovl_x_end], y=[_ovl_y_start, _ovl_y_end],
            mode='lines',
            line=dict(color='#1565c0', width=1.25, dash='dot'),
            name=f'{overlay_label} ({_ovl_ppy:.1f} pts/yr)' if overlay_label else f'Overlay ({_ovl_ppy:.1f} pts/yr)',
            hoverinfo='skip',
        ))

    today = _add_today_vline(fig)

    # --- Backtesting ---
    eci_is_backtesting = eci_proj_as_of_idx < len(tab_frontier_all) - 1
    eci_backtest_results = []
    _eci_bt_lookup = {}
    if eci_is_backtesting:
        _eci_bt_future = tab_frontier_all[eci_proj_as_of_idx + 1:]
        eci_backtest_results = _backtest_stats(
            _eci_bt_future, all_trajectories, eci_current['date'], proj_end_date,
            lambda m: m['eci_score'],
            lambda m: m['display_name'],
        )
        _eci_bt_lookup = {r['name']: r for r in eci_backtest_results}

    # --- Data points ---
    # Non-frontier models: only show those within 10 pts of frontier max to reduce clutter
    _eci_frontier_max = max(m['eci_score'] for m in tab_all if m['is_frontier'])
    _eci_nf_cutoff = _eci_frontier_max - 10
    for m in tab_all:
        if m['is_frontier'] or m['eci_score'] < _eci_nf_cutoff:
            continue
        hover = f"{m['display_name']}<br>{m['date'].strftime('%b %d, %Y')}<br>ECI: {m['eci_score']:.1f}"
        hover += _eci_gap_hover(m['eci_score'], m['date'])
        if eci_metr_proj:
            hover += _eci_metr_hover(m['eci_score'], m.get('organization', ''))
        fig.add_trace(go.Scatter(
            x=[m['date']], y=[m['eci_score']],
            mode='markers' + ('+text' if eci_show_labels else ''),
            marker=dict(color='#aaaaaa', size=6, symbol='circle-open',
                        line=dict(color='#bbbbbb', width=1)),
            text=[m['display_name']] if eci_show_labels else None,
            textposition='top right',
            textfont=dict(size=8, color='#bbbbbb'),
            hovertext=hover, hoverinfo='text', showlegend=False,
        ))

    # Then plot frontier models
    for idx_m, m in enumerate(tab_frontier_all):
        is_used = idx_m <= eci_proj_as_of_idx
        is_selected = idx_m == eci_proj_as_of_idx
        hover = f"{m['display_name']}<br>{m['date'].strftime('%b %d, %Y')}<br>ECI: {m['eci_score']:.1f}"
        hover += _eci_gap_hover(m['eci_score'], m['date'])
        if eci_metr_proj:
            hover += _eci_metr_hover(m['eci_score'], m.get('organization', ''))

        if is_used:
            color = '#e74c3c' if is_selected else '#4F8DFD'
            sym = 'star' if is_selected else 'circle'
            sz = 14 if is_selected else 10
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['eci_score']],
                mode='markers' + ('+text' if eci_show_labels else ''),
                marker=dict(color=color, size=sz, symbol=sym,
                            line=dict(color='white', width=1)),
                text=[m['display_name']] if eci_show_labels else None,
                textposition='top right',
                textfont=dict(size=9, color='#c0392b' if is_selected else '#1a1a2e'),
                hovertext=hover, hoverinfo='text', showlegend=False,
            ))
        else:
            _eci_bt_name = m['display_name']
            if eci_is_backtesting and _eci_bt_name in _eci_bt_lookup:
                r = _eci_bt_lookup[_eci_bt_name]
                _btc = _bt_color_for(r)
                _bt_label = f"{_eci_bt_name} (p{r['percentile']:.0f})"
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[m['eci_score']],
                    mode='markers+text',
                    marker=dict(color=_btc, size=12, symbol='diamond',
                                line=dict(color='white', width=1)),
                    text=[_bt_label],
                    textposition='top right',
                    textfont=dict(size=9, color=_btc),
                    hovertext=hover + f"<br>Percentile: {r['percentile']:.0f}%",
                    hoverinfo='text', showlegend=False,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[m['eci_score']],
                    mode='markers' + ('+text' if eci_show_labels else ''),
                    marker=dict(color='#aaaaaa', size=10, symbol='circle-open',
                                line=dict(color='#777777', width=2)),
                    text=[m['display_name']] if eci_show_labels else None,
                    textposition='top right',
                    textfont=dict(size=9, color='#999999'),
                    hovertext=hover, hoverinfo='text', showlegend=False,
                ))

    # --- Backtest overlay ---
    if eci_is_backtesting and eci_backtest_results:
        _add_backtest_traces(fig, eci_backtest_results, eci_current['date'])

    # --- Layout ---
    # Determine y range from data and projections
    all_scores = [m['eci_score'] for m in tab_all if m['is_frontier'] or m['eci_score'] >= _eci_nf_cutoff]
    y_min = min(all_scores) - 5
    _milestone_y_max = max(s for s, _, _ in milestone_list) if milestone_list else 170
    y_max = max(pct95[-1], max(all_scores) + 5, _milestone_y_max) + 5
    yaxis_cfg = dict(
        title="ECI Score",
        range=[y_min, y_max],
        gridcolor='rgba(0,0,0,0.1)',
        zeroline=False,
        tickfont=dict(color='#1a1a2e'),
        title_font=dict(color='#1a1a2e'),
    )

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=140, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(
            range=[tab_all[0]['date'] - timedelta(days=30),
                   proj_end_date + timedelta(days=30)],
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(color='#1a1a2e'),
            zeroline=False,
        ),
        yaxis=yaxis_cfg,
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)',
                    font=dict(color='#1a1a2e')),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # ── Render chart + metrics ──────────────────────────────────────────────
    st.plotly_chart(fig, width="stretch")
    if eci_is_backtesting and eci_backtest_results:
        _backtest_summary(eci_backtest_results)

    # ── Projections row ───────────────────────────────────────────────────

    eci_start_samples = eci_proj_start
    eci_current_label = eci_current['display_name']

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026EOY", datetime(2026, 12, 31)),
        ("2027 Jun EOM", datetime(2027, 6, 30)),
        ("2027EOY", datetime(2027, 12, 31)),
        ("2028EOY", datetime(2028, 12, 31)),
        ("2029EOY", datetime(2029, 12, 31)),
    ]

    def _proj_score_at(elapsed_days, start_scores, dpp_arr, superexp=False, hl=None, floor=None):
        """Project ECI score forward by elapsed_days."""
        if superexp and hl is not None:
            if floor is not None and floor > 0:
                t_cap = np.where(dpp_arr > floor, hl * np.log2(dpp_arr / floor), 0.0)
                se_phase = np.minimum(elapsed_days, t_cap)
                pts_se = (hl / (dpp_arr * np.log(2))) * (2**(se_phase / hl) - 1)
                pts_lin = np.maximum(elapsed_days - t_cap, 0) / floor
                pts = pts_se + pts_lin
            else:
                pts = (hl / (dpp_arr * np.log(2))) * (2**(elapsed_days / hl) - 1)
        else:
            pts = elapsed_days / dpp_arr
        return start_scores + pts

    # All columns use the projection model for coherent forecasts
    all_targets = [
        (f"{eci_current_label} ({eci_current['date'].strftime('%b %Y')})", eci_current['date']),
    ] + eoy_targets
    n_all_cols = len(all_targets)
    cols = st.columns([1.2] + [1] * (n_all_cols - 1))

    _eci_metr_a = 1 if eci_current.get('organization', '') == 'Anthropic' else 0

    def _metr_median_str(eci_score):
        p50_min = 2 ** (0.24 * eci_score + 0.76 * _eci_metr_a - 28.68)
        p50_lo, p50_hi = p50_min * 0.66, p50_min * 1.34
        return f"METR p50 {fmt_hrs(p50_lo / 60)}\u2013{fmt_hrs(p50_hi / 60)}"

    # When an overlay (US) frontier is supplied, the metric cards show how many
    # months China lags the US trend at that score rather than the raw ECI.
    _gap_trend = _eci_gap_fn

    for col, (label, target_date) in zip(cols, all_targets):
        elapsed = (target_date - eci_current['date']).days
        proj_scores = _proj_score_at(
            elapsed, eci_start_samples, eci_proj_dpp,
            eci_is_superexp, eci_superexp_halflife,
            eci_superexp_dpp_floor if eci_is_superexp else None)
        p10_s, p50_s, p90_s = np.percentile(proj_scores, [10, 50, 90])
        display_s = eci_current_score if elapsed == 0 else p50_s
        with col:
            if _gap_trend is not None:
                # Lower China score => further behind, so p10 gives the larger gap.
                g50 = _gap_trend(display_s, target_date)
                g_lo = _gap_trend(p90_s, target_date)
                g_hi = _gap_trend(p10_s, target_date)
                st.metric(label=label, value=f"{g50:.1f} mo")
                st.caption(f"80% CI: {g_lo:.1f} \u2013 {g_hi:.1f} mo")
            else:
                st.metric(label=label, value=f"{display_s:.1f}")
                st.caption(f"80% CI: {p10_s:.1f} \u2013 {p90_s:.1f}")
            if eci_metr_proj:
                st.markdown(
                    f"<div style='font-size:0.72em; color:#888; margin-top:-0.4em;'>"
                    f"{_metr_median_str(display_s)}</div>",
                    unsafe_allow_html=True,
                )

    # Milestone tables (extra_table_milestones appear only here, not as chart hlines)
    eci_milestone_thresholds = [(s, l) for s, l, _c in milestone_list]
    if extra_table_milestones:
        eci_milestone_thresholds = sorted(
            eci_milestone_thresholds + list(extra_table_milestones),
            key=lambda x: x[0])

    with st.expander("Milestone details"):
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown("**Probabilities**")
            rows = []
            for score_threshold, ms_label in eci_milestone_thresholds:
                row = {"Milestone": ms_label}
                for eoy_label, target_date in eoy_targets:
                    elapsed = (target_date - eci_current['date']).days
                    proj_scores = _proj_score_at(
                        elapsed, eci_start_samples, eci_proj_dpp,
                        eci_is_superexp, eci_superexp_halflife,
                        eci_superexp_dpp_floor if eci_is_superexp else None)
                    prob = np.mean(proj_scores >= score_threshold) * 100
                    row[eoy_label] = f"{prob:.0f}%"
                rows.append(row)
            st.table(rows)

        with tcol2:
            st.markdown("**Estimated arrival**")
            arrival_rows = []
            for score_threshold, ms_label in eci_milestone_thresholds:
                pts_needed = score_threshold - eci_start_samples
                if eci_is_superexp and eci_superexp_halflife is not None:
                    dpp_floor = eci_superexp_dpp_floor
                    t_cap = np.where(eci_proj_dpp > dpp_floor,
                                     eci_superexp_halflife * np.log2(eci_proj_dpp / dpp_floor), 0.0)
                    pts_at_cap = (eci_superexp_halflife / (eci_proj_dpp * np.log(2))) * (2**(t_cap / eci_superexp_halflife) - 1)
                    # If needed points fit in superexp phase
                    arg = 1 + pts_needed * eci_proj_dpp * np.log(2) / eci_superexp_halflife
                    arg = np.maximum(arg, 1e-10)
                    days_se_only = eci_superexp_halflife * np.log2(arg)
                    # If not, use cap + linear remainder
                    leftover = np.maximum(pts_needed - pts_at_cap, 0)
                    days_with_floor = t_cap + leftover * dpp_floor
                    days_to = np.where(pts_needed <= pts_at_cap, days_se_only, days_with_floor)
                else:
                    days_to = pts_needed * eci_proj_dpp
                # Filter out negative/zero days (already past milestone)
                days_to = np.maximum(days_to, 0)
                p10_d, p50_d, p90_d = np.percentile(days_to, [10, 50, 90])
                med_date = eci_current['date'] + timedelta(days=max(p50_d, 0))
                early_date = eci_current['date'] + timedelta(days=max(p10_d, 0))
                late_date = eci_current['date'] + timedelta(days=max(p90_d, 0))
                arrival_rows.append({
                    "Milestone": ms_label,
                    "Median": med_date.strftime('%b %Y'),
                    "80% CI": f"{early_date.strftime('%b %Y')} \u2013 {late_date.strftime('%b %Y')}",
                })
            st.table(arrival_rows)

    _fn_caption(
        "ECI = Epoch Capabilities Index. +Pts/Yr = ECI points gained per year. "
        "These are projections, not forecasts.",
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))



# ── Company registry: single source of truth for BOTH ECI tabs ──────────
#
# The Epoch ECI tab and the ECI Company Gap tab both need to know which of
# Epoch's `Organization` strings belong to which company. That knowledge used to
# live twice -- as `orgs` substring lists here and as an `_ECG_ORG_MAP` keyed the
# other way round in the gap tab -- and the two drifted: the gap tab dropped four
# Google models because Epoch spells Google four ways and its map was keyed only
# on "Google DeepMind". Everything both tabs need is now derived from this one
# table, so they cannot disagree by construction.
#
#   orgs    -- substrings matched case-insensitively against `Organization`,
#              which handles Epoch's comma-joined multi-org spellings. Keep them
#              MINIMAL: substring matching makes longer variants redundant
#              ("Google" already catches "Google DeepMind,Google"). Adding a
#              redundant variant is harmless but adds a thing to keep in sync.
#   country -- key into _ECG_FLAG.
#   color   -- gap-tab highlight colour.
#   slug    -- ?a= / ?b= deep-link slug on the ECI tab.
#
# Insertion order is the ECI tab's dropdown order and is load-bearing: the
# country entities must come first so `US best` stays the index-0 default.
#
# Substring matching is only unambiguous while no `Organization` string contains
# two different companies. `TestEcgOrgMatching` asserts that across every
# distinct string in the CSV, so a future Epoch pull that breaks it fails loudly
# rather than silently misattributing models.
_ECI_COMPANIES = {
    "Anthropic":         {"orgs": ["Anthropic"], "country": "US", "color": "#d4a574", "slug": "anthropic"},
    "OpenAI":            {"orgs": ["OpenAI"],    "country": "US", "color": "#10a37f", "slug": "openai"},
    "xAI":               {"orgs": ["xAI"],       "country": "US", "color": "#555555", "slug": "xai"},
    "Google":            {"orgs": ["Google"],    "country": "US", "color": "#4285F4", "slug": "google"},
    "Meta":              {"orgs": ["Meta AI"],   "country": "US", "color": "#0668E1", "slug": "meta"},
    "Alibaba":           {"orgs": ["Alibaba"],   "country": "CN", "color": "#E74C3C", "slug": "alibaba"},
    "Zhipu AI":          {"orgs": ["Zhipu"],     "country": "CN", "color": "#2ECC71", "slug": "zhipu"},
    "Moonshot":          {"orgs": ["Moonshot"],  "country": "CN", "color": "#9B59B6", "slug": "moonshot"},
    "DeepSeek":          {"orgs": ["DeepSeek"],  "country": "CN", "color": "#1e90ff", "slug": "deepseek"},
    "Mistral":           {"orgs": ["Mistral"],   "country": "FR", "color": "#FF7000", "slug": "mistral"},
    "MiniMax":           {"orgs": ["MiniMax"],   "country": "CN", "color": "#16A085", "slug": "minimax"},
    "Thinking Machines": {"orgs": ["Thinking Machines"], "country": "US", "color": "#C0392B",
                          "slug": "thinkingmachines"},
}

# The "best" aggregates are country filters, not companies, so they stay explicit.
# `country` here is the full CSV value, unlike _ECI_COMPANIES' two-letter flag key.
_ECI_COUNTRY_ENTITIES = {
    "US best":    {"country": "United States of America", "slug": "us"},
    "China best": {"country": "China", "slug": "china"},
}

# Entities selectable in the two ECI comparison dropdowns. Each label maps to a
# filter over the ECI model list: a country for the "best" aggregates, or a list
# of Organization substrings for an individual lab.
_ECI_ENTITY_SPECS = {
    **{n: {"country": c["country"]} for n, c in _ECI_COUNTRY_ENTITIES.items()},
    **{n: {"orgs": c["orgs"]} for n, c in _ECI_COMPANIES.items()},
}
_ECI_ENTITY_OPTIONS = list(_ECI_ENTITY_SPECS.keys())
_ECI_NONE_LABEL = "—"  # em-dash: "no comparison" for the second dropdown

# Absolute milestone lines shown for the default US-best frontier.
_ECI_US_MILESTONES = [
    (155, "ECI 155", '#888888'), (160, "ECI 160", '#666666'),
    (165, "ECI 165", '#c0392b'), (170, "ECI 170", '#8e44ad'),
]

# Deep-link slugs for ?a= / ?b= URL params, derived from the registry above.
_ECI_ENTITY_SLUG = {
    **{n: c["slug"] for n, c in _ECI_COUNTRY_ENTITIES.items()},
    **{n: c["slug"] for n, c in _ECI_COMPANIES.items()},
}
_ECI_ENTITY_FOR_SLUG = {v: k for k, v in _ECI_ENTITY_SLUG.items()}


def _eci_entity_short(label):
    """'US best' → 'US', so composed labels don't read 'US best best 148.3'."""
    return label[:-len(" best")] if label.endswith(" best") else label


def _eci_entity_data(label):
    """Return (all, frontier_all, frontier_names) for a comparison entity."""
    spec = _ECI_ENTITY_SPECS[label]
    all_ = load_eci_frontier(
        _mtime=_eci_mtime(), country=spec.get("country"),
        orgs=tuple(spec["orgs"]) if spec.get("orgs") else None)
    fr = [m for m in all_ if m['is_frontier']]
    return all_, fr, [m['display_name'] for m in fr]


def render_eci():
    # Two dropdowns. The first is the benchmark/reference (default "US best");
    # the second is the subject projected against it (default none). With no
    # second entity, the benchmark itself is projected as a plain single view.
    # So [US, blank] = US frontier; [US, China] = China projected vs the US
    # trend (the old China view), reading naturally left-to-right.
    # Deep-link via ?a= (benchmark) / ?b= (subject); omit at the defaults.
    _b_opts = [_ECI_NONE_LABEL] + _ECI_ENTITY_OPTIONS
    _url_a = _ECI_ENTITY_FOR_SLUG.get(st.query_params.get("a", "").lower())
    _url_b = _ECI_ENTITY_FOR_SLUG.get(st.query_params.get("b", "").lower())
    # Back-compat: the old ?region=china deep link maps to US benchmark / China.
    if _url_a is None and st.query_params.get("region", "").lower() == "china":
        _url_a, _url_b = "US best", "China best"
    _a_idx = _ECI_ENTITY_OPTIONS.index(_url_a) if _url_a in _ECI_ENTITY_OPTIONS else 0
    _b_idx = _b_opts.index(_url_b) if _url_b in _b_opts else 0

    with st.sidebar:
        entity_a = st.selectbox(
            "Entity", _ECI_ENTITY_OPTIONS, index=_a_idx, key="eci_entity_a")
        entity_b = st.selectbox(
            "Compare to", _b_opts, index=_b_idx, key="eci_entity_b")

    # Keep the URL in sync (omit params at their defaults).
    if entity_a == "US best":
        st.query_params.pop("a", None)
    else:
        st.query_params["a"] = _ECI_ENTITY_SLUG[entity_a]
    if entity_b == _ECI_NONE_LABEL:
        st.query_params.pop("b", None)
    else:
        st.query_params["b"] = _ECI_ENTITY_SLUG[entity_b]
    st.query_params.pop("region", None)  # superseded by ?a= / ?b=

    # Short display names ("US best" → "US") for composed labels/titles.
    _a_name = _eci_entity_short(entity_a)
    _b_name = _eci_entity_short(entity_b)

    # No (or self-) comparison: project the benchmark itself, single-entity view.
    if entity_b == _ECI_NONE_LABEL or entity_b == entity_a:
        s_all, s_fr, s_names = _eci_entity_data(entity_a)
        if len(s_fr) < 2:
            st.warning(f"Not enough {_a_name} models on the ECI frontier to project.")
            return
        milestones = _ECI_US_MILESTONES if entity_a == "US best" else []
        _render_eci_tab(
            s_all, s_fr, s_names, "eci",
            f"{_a_name} ECI Projection", milestones)
        return

    # Comparison view: project the subject (B), overlay the benchmark (A) trend.
    bench_all, bench_fr, bench_names = _eci_entity_data(entity_a)
    subj_all, subj_fr, subj_names = _eci_entity_data(entity_b)
    if len(subj_fr) < 2:
        st.warning(f"Not enough {_b_name} models on the ECI frontier to project.")
        return
    if len(bench_fr) < 2:
        st.warning(f"Not enough {_a_name} models to draw a comparison trend.")
        _render_eci_tab(subj_all, subj_fr, subj_names, "eci",
                        f"{_b_name} ECI Projection", [])
        return

    # Benchmark's best model → milestone line + labeled dot at its release date.
    bench_best = max(bench_fr, key=lambda m: m['eci_score'])
    subj_current = max(subj_fr, key=lambda m: m['eci_score'])
    # The benchmark model nearest the subject's current best marks where the
    # benchmark passed the level the subject sits at today. Only meaningful when
    # the benchmark actually reached that level (it leads); if the subject leads,
    # there's no such crossing, so omit the marker.
    bench_match = None
    if bench_best['eci_score'] >= subj_current['eci_score']:
        bench_match = min(
            bench_fr, key=lambda m: abs(m['eci_score'] - subj_current['eci_score']))
    _render_eci_tab(
        subj_all, subj_fr, subj_names, "eci",
        f"{_b_name} ECI Projection",
        [(bench_best['eci_score'], f"{_a_name} best {bench_best['eci_score']:.1f}", '#8e44ad')],
        overlay_frontier=bench_fr,
        overlay_label=f"{_a_name} trend",
        overlay_name=_a_name,
        us_best_marker={
            'date': bench_best['date'],
            'score': bench_best['eci_score'],
            'name': bench_best['display_name'],
        },
        us_match_marker=None if bench_match is None else {
            'date': bench_match['date'],
            'score': bench_match['eci_score'],
            'name': bench_match['display_name'],
        },
        us_match_label=(None if bench_match is None
                        else f"{_b_name} best {subj_current['eci_score']:.1f}"),
        subject_name=_b_name,
    )


# ── Remote Labor Index ───────────────────────────────────────────────────

_RLI_RESET_KEYS = [
    "rli_custom_dt_lo", "rli_custom_dt_hi",
    "rli_custom_pos_lo", "rli_custom_pos_hi",
    "rli_piecewise_n_seg", "rli_bp1_select",
    "rli_bp2_select", "rli_custom_dt_dist",
    "rli_custom_pos_dist",
    "rli_superexp_dt_init", "rli_superexp_halflife",
    "rli_superexp_dt_floor", "rli_superexp_dt_ci_lo",
    "rli_superexp_dt_ci_hi", "rli_superexp_pos_lo",
    "rli_superexp_pos_hi",
    "rli_proj_basis", "rli_milestones", "rli_labels",
    "rli_log_scale", "_rli_proj_as_of", "rli_end_year",
    "_rli_seg_config",
]

_RLI_DEFAULTS = {
    "rli_proj_basis": "Linear (logit)",
    "rli_piecewise_n_seg": 1,
    "rli_custom_dt_dist": "Lognormal",
    "rli_custom_pos_dist": "Normal",
    "rli_milestones": True,
    "rli_labels": True,
    "rli_log_scale": False,
    "rli_end_year": 2026,
}

def render_rli():
    if st.session_state.pop("_reset_rli", False):
        for k in _RLI_RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(_RLI_DEFAULTS)
        st.rerun()

    for k, v in _RLI_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── RLI Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.header("RLI Projection")

        # Read "project as of" from session state
        rli_proj_as_of_name = st.session_state.get('_rli_proj_as_of', rli_frontier_names[-1])
        if rli_proj_as_of_name not in rli_frontier_names:
            rli_proj_as_of_name = rli_frontier_names[-1]
        rli_proj_as_of_idx = rli_frontier_names.index(rli_proj_as_of_name)

        # --- Projection basis ---
        rli_basis_options = ["Linear (logit)", "Piecewise linear (logit)", "Superexponential (logit)"]
        rli_proj_basis = st.radio("Projection basis", rli_basis_options, key="rli_proj_basis",
                                  help="All projections use logit-space fitting to keep scores bounded 0–100%.")

        rli_custom_dt_lo = rli_custom_dt_hi = None
        rli_custom_pos_lo = rli_custom_pos_hi = None
        rli_custom_dt_dist = "Lognormal"
        rli_custom_pos_dist = "Normal"
        rli_piecewise_n_segments = 1
        rli_piecewise_breakpoints = []
        _rli_is_linear = rli_proj_basis in ("Linear (logit)", "Piecewise linear (logit)")
        if rli_proj_basis == "Piecewise linear (logit)":
            rli_piecewise_n_segments = 2

        # Pre-compute OLS DT for data-driven defaults
        _rli_pre_fr = rli_frontier_all[:rli_proj_as_of_idx + 1]
        _rli_pre_base = rli_frontier_all[0]['date']
        _rli_pre_days = np.array([(m['date'] - _rli_pre_base).days for m in _rli_pre_fr], dtype=float)
        _rli_pre_logit = _logit(np.array([m['rli_score'] / 100 for m in _rli_pre_fr]))
        _rli_pre_params = fit_line(_rli_pre_days, _rli_pre_logit) if len(_rli_pre_fr) >= 2 else np.array([0, 0.007])
        _rli_pre_dt = round(np.log(2) / _rli_pre_params[1]) if _rli_pre_params[1] > 0 else 100

        if _rli_is_linear:
            with st.expander("Advanced options"):
                st.button("Reset to defaults", key="reset_rli_linear",
                          on_click=lambda: st.session_state.update(_reset_rli=True))

                # Segments & breakpoints only for Piecewise linear (logit)
                _rli_bp_names = [m['name'] for m in rli_frontier_all[:rli_proj_as_of_idx + 1]]
                if rli_proj_basis == "Piecewise linear (logit)":
                    _rli_seg_options = [1, 2, 3] if len(_rli_bp_names) >= 5 else [1, 2]
                    if rli_piecewise_n_segments not in _rli_seg_options:
                        rli_piecewise_n_segments = _rli_seg_options[-1]
                    # Ensure session state defaults to 2 for Piecewise
                    if st.session_state.get("rli_piecewise_n_seg", 1) < 2:
                        st.session_state["rli_piecewise_n_seg"] = 2
                    rli_piecewise_n_segments = st.radio(
                        "Segments", _rli_seg_options,
                        horizontal=True, key="rli_piecewise_n_seg")
                else:
                    # Plain Linear: force 1 segment, clear stale session state
                    rli_piecewise_n_segments = 1
                    st.session_state.pop("rli_piecewise_n_seg", None)
                if rli_piecewise_n_segments >= 2:
                    _rli_default_bp1 = _rli_bp_names[len(_rli_bp_names) // 2]
                    _rli_bp1_idx = _rli_bp_names.index(_rli_default_bp1) if _rli_default_bp1 in _rli_bp_names else len(_rli_bp_names) // 2
                    rli_bp1_name = st.selectbox(
                        "Breakpoint", _rli_bp_names[1:],
                        index=max(0, _rli_bp1_idx - 1), key="rli_bp1_select")
                    rli_piecewise_breakpoints.append(rli_bp1_name)
                if rli_piecewise_n_segments >= 3:
                    _rli_bp1_pos = _rli_bp_names.index(rli_bp1_name)
                    _rli_remaining = _rli_bp_names[_rli_bp1_pos + 1:]
                    rli_bp2_name = st.selectbox(
                        "Breakpoint 2", _rli_remaining[:-1],
                        index=len(_rli_remaining[:-1]) // 2, key="rli_bp2_select")
                    rli_piecewise_breakpoints.append(rli_bp2_name)

                # Compute DT defaults from the actual last segment
                if rli_piecewise_n_segments >= 2 and rli_piecewise_breakpoints:
                    _rli_last_bp_idx = _rli_bp_names.index(rli_piecewise_breakpoints[-1]) if rli_piecewise_breakpoints[-1] in _rli_bp_names else 0
                    _rli_pw_seg_days = _rli_pre_days[_rli_last_bp_idx:]
                    _rli_pw_seg_logit = _rli_pre_logit[_rli_last_bp_idx:]
                    if len(_rli_pw_seg_days) >= 2:
                        _rli_pw_seg_params = fit_line(_rli_pw_seg_days, _rli_pw_seg_logit)
                        _rli_pw_seg_dt = round(np.log(2) / _rli_pw_seg_params[1]) if _rli_pw_seg_params[1] > 0 else _rli_pre_dt
                    else:
                        _rli_pw_seg_dt = _rli_pre_dt
                    _rli_default_dt_lo = float(round(max(5.0, _rli_pw_seg_dt / 2), 0))
                    _rli_default_dt_hi = float(round(_rli_pw_seg_dt * 2, 0))
                else:
                    _rli_default_dt_lo = float(round(max(5.0, _rli_pre_dt / 2), 0))
                    _rli_default_dt_hi = float(round(_rli_pre_dt * 2, 0))

                # Auto-update DT CIs when segment config changes
                _rli_seg_config = (rli_piecewise_n_segments, tuple(rli_piecewise_breakpoints))
                if st.session_state.get("_rli_seg_config") != _rli_seg_config:
                    st.session_state["_rli_seg_config"] = _rli_seg_config
                    st.session_state.pop("rli_custom_dt_lo", None)
                    st.session_state.pop("rli_custom_dt_hi", None)

                # Doubling time CI (days for odds to double)
                _rli_dt_lo_col, _rli_dt_hi_col = st.columns(2)
                rli_custom_dt_lo = _ss_number_input(_rli_dt_lo_col,
                    "Odds 2x time CI low (days)", "rli_custom_dt_lo", _rli_default_dt_lo,
                    min_value=5.0, max_value=2000.0, step=5.0,
                    help="Fast scenario: days for odds p/(1-p) to double.")
                rli_custom_dt_hi = _ss_number_input(_rli_dt_hi_col,
                    "Odds 2x time CI high (days)", "rli_custom_dt_hi", _rli_default_dt_hi,
                    min_value=5.0, max_value=5000.0, step=5.0,
                    help="Slow scenario: days for odds to double.")
                if rli_custom_dt_lo > rli_custom_dt_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()

                # Position CI in percentage points
                _rli_cur = rli_frontier_all[rli_proj_as_of_idx]
                _rli_def_score = _rli_cur['rli_score']
                _rli_pos_lo_col, _rli_pos_hi_col = st.columns(2)
                rli_custom_pos_lo = _ss_number_input(_rli_pos_lo_col,
                    "Pos CI low (%)", "rli_custom_pos_lo", round(max(_rli_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1)
                rli_custom_pos_hi = _ss_number_input(_rli_pos_hi_col,
                    "Pos CI high (%)", "rli_custom_pos_hi", round(_rli_def_score + 1.0, 2),
                    step=0.1)

                rli_custom_dt_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"],
                    horizontal=True, key="rli_custom_dt_dist")
                rli_custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal"],
                    horizontal=True, key="rli_custom_pos_dist")

        # --- Superexponential controls ---
        rli_superexp_dt_initial = rli_superexp_halflife = None
        rli_superexp_dt_ci_lo = rli_superexp_dt_ci_hi = None
        rli_superexp_pos_lo = rli_superexp_pos_hi = None
        rli_superexp_dt_floor = 10
        rli_is_superexp = False
        if rli_proj_basis == "Superexponential (logit)":
            rli_is_superexp = True
            _rli_default_dt_init = 100.0
            if len(rli_frontier_all[:rli_proj_as_of_idx + 1]) >= 2:
                _rli_base = rli_frontier_all[0]['date']
                _rli_fr = rli_frontier_all[:rli_proj_as_of_idx + 1]
                _rli_fd = np.array([(m['date'] - _rli_base).days for m in _rli_fr], dtype=float)
                _rli_flogit = _logit(np.array([m['rli_score'] / 100 for m in _rli_fr]))
                _rli_fp = fit_line(_rli_fd, _rli_flogit)
                if _rli_fp[1] > 0:
                    _rli_default_dt_init = round(np.log(2) / _rli_fp[1], 0)

            # Pre-compute superexp fit at default halflife for CI defaults
            _rli_pre_se_halflife = 365
            _rli_pre_se_z = 2 ** (_rli_pre_days / _rli_pre_se_halflife)
            _rli_pre_se_X = np.column_stack([np.ones_like(_rli_pre_se_z), _rli_pre_se_z])
            (_rli_pre_se_A, _rli_pre_se_K), *_ = np.linalg.lstsq(_rli_pre_se_X, _rli_pre_logit, rcond=None)
            _rli_pre_se_d_last = _rli_pre_days[-1]
            if _rli_pre_se_K > 0:
                _rli_pre_se_logit_slope = _rli_pre_se_K * np.log(2) * 2 ** (_rli_pre_se_d_last / _rli_pre_se_halflife) / _rli_pre_se_halflife
                _rli_pre_se_dt = round(np.log(2) / _rli_pre_se_logit_slope, 0)
            else:
                _rli_pre_se_dt = _rli_pre_dt
            _rli_default_se_dt_lo = float(round(max(5.0, _rli_pre_se_dt / 2), 0))
            _rli_default_se_dt_hi = float(round(_rli_pre_se_dt * 2, 0))

            with st.expander("Advanced options"):
                st.button("Reset to defaults", key="reset_rli_superexp",
                          on_click=lambda: st.session_state.update(_reset_rli=True))
                _rli_se_col1, _rli_se_col2 = st.columns(2)
                rli_superexp_dt_initial = _ss_number_input(_rli_se_col1,
                    "Initial odds 2x time (days)", "rli_superexp_dt_init", _rli_default_dt_init,
                    min_value=5.0, max_value=2000.0, step=5.0)
                rli_superexp_halflife = _ss_number_input(_rli_se_col2,
                    "Rate half-life (days)", "rli_superexp_halflife", 365,
                    min_value=30, max_value=5000, step=30,
                    help="How quickly rate grows. Lower = faster.")
                rli_superexp_dt_floor_input = _ss_number_input(st,
                    "Min odds 2x time (days)", "rli_superexp_dt_floor", 15.0,
                    min_value=1.0, max_value=500.0, step=1.0,
                    help="Rate can't exceed this. Prevents runaway projections.")
                rli_superexp_dt_floor = rli_superexp_dt_floor_input
                _rli_se_ci1, _rli_se_ci2 = st.columns(2)
                rli_superexp_dt_ci_lo = _ss_number_input(_rli_se_ci1,
                    "Odds 2x CI low (days)", "rli_superexp_dt_ci_lo", _rli_default_se_dt_lo,
                    min_value=5.0, max_value=2000.0, step=5.0)
                rli_superexp_dt_ci_hi = _ss_number_input(_rli_se_ci2,
                    "Odds 2x CI high (days)", "rli_superexp_dt_ci_hi", _rli_default_se_dt_hi,
                    min_value=5.0, max_value=5000.0, step=5.0)
                if rli_superexp_dt_ci_lo > rli_superexp_dt_ci_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()
                _rli_cur = rli_frontier_all[rli_proj_as_of_idx]
                _rli_def_score = _rli_cur['rli_score']
                _rli_se_pos1, _rli_se_pos2 = st.columns(2)
                rli_superexp_pos_lo = _ss_number_input(_rli_se_pos1,
                    "Pos CI low (%)", "rli_superexp_pos_lo", round(max(_rli_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1)
                rli_superexp_pos_hi = _ss_number_input(_rli_se_pos2,
                    "Pos CI high (%)", "rli_superexp_pos_hi", round(_rli_def_score + 1.0, 2),
                    step=0.1)

        st.markdown("---")
        rli_show_milestones = st.toggle("Milestones", key="rli_milestones")
        rli_show_labels = st.toggle("Labels", key="rli_labels")
        rli_use_log_scale = st.toggle("Log scale", key="rli_log_scale")

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                rli_frontier_names,
                index=rli_frontier_names.index(rli_proj_as_of_name),
                key='_rli_proj_as_of',
                help="Backtest: project from an earlier model's vantage point.",
            )
            _rli_end_year = st.radio(
                "Project through", [2026, 2027, 2028, 2029],
                horizontal=True, key="rli_end_year")

    # ── Build data arrays ────────────────────────────────────────────────────
    rli_frontier_used = rli_frontier_all[:rli_proj_as_of_idx + 1]

    base_date = rli_frontier_all[0]['date']
    days_all_rli = np.array([(m['date'] - base_date).days for m in rli_frontier_all], dtype=float)
    scores_all_rli = np.array([m['rli_score'] for m in rli_frontier_all])
    logit_all_rli = _logit(scores_all_rli / 100)

    _rli_fit_start = 0
    _rli_fit_end = rli_proj_as_of_idx + 1
    rli_frontier_used = rli_frontier_all[_rli_fit_start:_rli_fit_end]
    days_used = days_all_rli[_rli_fit_start:_rli_fit_end]
    logit_used = logit_all_rli[_rli_fit_start:_rli_fit_end]
    n_used = len(rli_frontier_used)

    # Doubling time of odds: dt = ln(2) / logit_slope_per_day
    # logit_slope = ln(2) / dt

    if rli_proj_basis in ("Linear (logit)", "Piecewise linear (logit)"):
        if rli_piecewise_n_segments >= 2:
            _rli_bp_names_used = [m['name'] for m in rli_frontier_used]
            _rli_seg_break_idxs = []
            for bp_name in rli_piecewise_breakpoints:
                if bp_name in _rli_bp_names_used:
                    _rli_seg_break_idxs.append(_rli_bp_names_used.index(bp_name))
            _rli_last_seg_start = _rli_seg_break_idxs[-1] if _rli_seg_break_idxs else 0
            _rli_last_seg_range = list(range(_rli_last_seg_start, n_used))
            _rli_params = fit_line(days_used[_rli_last_seg_range], logit_used[_rli_last_seg_range])
        else:
            _rli_params = fit_line(days_used, logit_used)

        _rli_current_day = (rli_frontier_used[-1]['date'] - base_date).days
        if rli_piecewise_n_segments >= 2:
            _rli_seg_d = days_used[_rli_last_seg_range]
            _rli_seg_y = logit_used[_rli_last_seg_range]
        else:
            _rli_seg_d = days_used
            _rli_seg_y = logit_used
        _rli_intercept = np.mean(_rli_seg_y - _rli_params[1] * _rli_seg_d)
        _rli_fitted_logit = _rli_intercept + _rli_params[1] * _rli_current_day

        n_rli = N_SAMPLES
        if rli_custom_dt_dist == "Log-log":
            rli_proj_dt = _log_lognormal_from_ci(rli_custom_dt_lo, rli_custom_dt_hi, n_rli)
        elif rli_custom_dt_dist == "Lognormal":
            rli_proj_dt = _lognormal_from_ci(rli_custom_dt_lo, rli_custom_dt_hi, n_rli)
        else:
            rli_proj_dt = _normal_from_ci(rli_custom_dt_lo, rli_custom_dt_hi, n_rli)

        # Convert doubling times to logit slopes: slope = ln(2) / dt
        rli_proj_logit_slope = np.log(2) / rli_proj_dt

        # Position samples in logit space
        if rli_custom_pos_dist == "Lognormal":
            _rli_pos_logit_lo = _logit(rli_custom_pos_lo / 100)
            _rli_pos_logit_hi = _logit(rli_custom_pos_hi / 100)
            _rli_pos_offset = 10  # shift so values are safely positive
            _rli_pos_sigma = (np.log(_rli_pos_logit_hi + _rli_pos_offset) - np.log(_rli_pos_logit_lo + _rli_pos_offset)) / (2 * 1.282)
            _rli_pos_mu = np.log(_rli_fitted_logit + _rli_pos_offset)
            rli_proj_start_logit = np.random.lognormal(_rli_pos_mu, max(_rli_pos_sigma, 0), n_rli) - _rli_pos_offset
        else:
            _rli_pos_logit_lo = _logit(rli_custom_pos_lo / 100)
            _rli_pos_logit_hi = _logit(rli_custom_pos_hi / 100)
            _rli_pos_sigma = (_rli_pos_logit_hi - _rli_pos_logit_lo) / (2 * 1.282)
            rli_proj_start_logit = np.random.normal(_rli_fitted_logit, max(_rli_pos_sigma, 0), n_rli)

    elif rli_proj_basis == "Superexponential (logit)":
        # In logit space: logit = A + K * 2^(d/halflife)
        _rli_se_days = np.array([(m['date'] - base_date).days for m in rli_frontier_used], dtype=float)
        _rli_se_logit = _logit(np.array([m['rli_score'] / 100 for m in rli_frontier_used]))
        _rli_se_z = 2 ** (_rli_se_days / rli_superexp_halflife)
        _rli_se_X = np.column_stack([np.ones_like(_rli_se_z), _rli_se_z])
        (_rli_se_A, _rli_se_K), *_ = np.linalg.lstsq(_rli_se_X, _rli_se_logit, rcond=None)

        _rli_se_current_day = (rli_frontier_used[-1]['date'] - base_date).days
        _rli_se_fitted_logit = _rli_se_A + _rli_se_K * 2 ** (_rli_se_current_day / rli_superexp_halflife)

        # Implied doubling time at current date
        if _rli_se_K > 0:
            _rli_se_logit_slope = _rli_se_K * np.log(2) * 2 ** (_rli_se_current_day / rli_superexp_halflife) / rli_superexp_halflife
            rli_superexp_dt_fitted = np.log(2) / _rli_se_logit_slope
        else:
            rli_superexp_dt_fitted = float('inf')

        n_rli = N_SAMPLES
        rli_proj_dt = _lognormal_from_ci(rli_superexp_dt_ci_lo, rli_superexp_dt_ci_hi, n_rli)
        rli_proj_logit_slope = np.log(2) / rli_proj_dt

        # Position: normal noise in logit space
        _rli_se_pos_logit_lo = _logit(rli_superexp_pos_lo / 100)
        _rli_se_pos_logit_hi = _logit(rli_superexp_pos_hi / 100)
        _rli_se_pos_sigma = (_rli_se_pos_logit_hi - _rli_se_pos_logit_lo) / (2 * 1.282)
        rli_proj_start_logit = np.random.normal(_rli_se_fitted_logit, max(_rli_se_pos_sigma, 0), n_rli)

    # ── Current SOTA ──────────────────────────────────────────────────────
    rli_current = rli_frontier_used[-1]
    rli_current_score = rli_current['rli_score']

    # ── Build trajectories ────────────────────────────────────────────────
    proj_end_date = datetime(_rli_end_year, 12, 31)
    proj_n_days = (proj_end_date - rli_current['date']).days + 1
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates = [rli_current['date'] + timedelta(days=int(d)) for d in proj_days_arr]

    n_samples = len(rli_proj_dt)
    if rli_is_superexp:
        all_logit_traj = rli_proj_start_logit[:, None] + np.log(2) * superexp_trajectory(
            proj_days_arr, rli_proj_dt, rli_superexp_halflife, rli_superexp_dt_floor)
    else:
        all_logit_traj = rli_proj_start_logit[:, None] + proj_days_arr[None, :] * rli_proj_logit_slope[:, None]

    # Convert to percentage space
    all_trajectories = _inv_logit(all_logit_traj) * 100

    pct5 = np.percentile(all_trajectories, 5, axis=0)
    pct10 = np.percentile(all_trajectories, 10, axis=0)
    pct25 = np.percentile(all_trajectories, 25, axis=0)
    pct50 = np.percentile(all_trajectories, 50, axis=0)
    pct75 = np.percentile(all_trajectories, 75, axis=0)
    pct90 = np.percentile(all_trajectories, 90, axis=0)
    pct95 = np.percentile(all_trajectories, 95, axis=0)

    fig = go.Figure()

    # --- Fan bands ---
    bands_spec = [
        (pct5, pct95, 'rgba(52,152,219,0.10)', '90% CI'),
        (pct10, pct90, 'rgba(52,152,219,0.18)', '80% CI'),
        (pct25, pct75, 'rgba(52,152,219,0.28)', '50% CI'),
    ]
    for lo, hi, color, label in bands_spec:
        x_poly = proj_dates + proj_dates[::-1]
        y_poly = list(hi) + list(lo[::-1])
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly,
            fill='toself', fillcolor=color,
            line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True,
        ))

    # --- Trend line (in logit space, converted back) ---
    if rli_proj_basis in ("Linear (logit)", "Piecewise linear (logit)"):
        _seg_colors = ['#e74c3c', '#f39c12', '#27ae60']
        if rli_piecewise_n_segments >= 2:
            _rli_bp_names_used = [m['name'] for m in rli_frontier_used]
            _rli_break_idxs = []
            for bp_name in rli_piecewise_breakpoints:
                if bp_name in _rli_bp_names_used:
                    _rli_break_idxs.append(_rli_bp_names_used.index(bp_name))
            _rli_seg_bounds = [0] + _rli_break_idxs + [n_used]
            _rli_segments = []
            for si in range(len(_rli_seg_bounds) - 1):
                end = _rli_seg_bounds[si + 1] + 1 if si < len(_rli_seg_bounds) - 2 else _rli_seg_bounds[si + 1]
                _rli_segments.append(list(range(_rli_seg_bounds[si], min(end, n_used))))
            for si, seg_idx in enumerate(_rli_segments):
                if len(seg_idx) < 2:
                    continue
                seg_params = fit_line(days_used[seg_idx], logit_used[seg_idx])
                seg_dt = np.log(2) / seg_params[1] if seg_params[1] > 0 else float('inf')
                is_last = (si == len(_rli_segments) - 1)
                if is_last:
                    # Historical portion: OLS through data points
                    d0 = int(days_used[seg_idx[0]])
                    d_last = int(days_used[seg_idx[-1]])
                    days_range = np.arange(d0, d_last + 1, 1)
                    logit_trend = seg_params[0] + seg_params[1] * days_range
                    y_trend = _inv_logit(logit_trend) * 100
                    dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
                    hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_seg, y_trend)]
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_trend.tolist(),
                        mode='lines', line=dict(color='#2c3e50', width=2.5),
                        name=f'Segment {si+1} (2x odds: {seg_dt:.0f}d)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
                    # Projected portion: user DT slope from last data point
                    _user_dt_center = np.sqrt(rli_custom_dt_lo * rli_custom_dt_hi)
                    _user_logit_slope = np.log(2) / _user_dt_center
                    _ols_logit_at_last = seg_params[0] + seg_params[1] * d_last
                    _proj_intercept = _ols_logit_at_last - _user_logit_slope * d_last
                    d1 = (proj_end_date - base_date).days
                    days_proj = np.arange(d_last, d1 + 1, 1)
                    logit_proj = _proj_intercept + _user_logit_slope * days_proj
                    y_proj = _inv_logit(logit_proj) * 100
                    dates_proj = [base_date + timedelta(days=int(d)) for d in days_proj]
                    hover_proj = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_proj, y_proj)]
                    fig.add_trace(go.Scatter(
                        x=dates_proj, y=y_proj.tolist(),
                        mode='lines', line=dict(color='#2980b9', width=2.5),
                        name=f'Projection (2x odds: {_user_dt_center:.0f}d, CI {rli_custom_dt_lo}\u2013{rli_custom_dt_hi}d)',
                        hovertext=hover_proj, hoverinfo='text',
                    ))
                else:
                    d0 = int(days_used[seg_idx[0]])
                    d1 = int(days_used[seg_idx[-1]])
                    days_range = np.arange(d0, d1 + 1, 1)
                    logit_trend = seg_params[0] + seg_params[1] * days_range
                    y_trend = _inv_logit(logit_trend) * 100
                    dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
                    hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_seg, y_trend)]
                    fig.add_trace(go.Scatter(
                        x=dates_seg, y=y_trend.tolist(),
                        mode='lines', line=dict(color=_seg_colors[si % len(_seg_colors)], width=2, dash='dash'),
                        name=f'Segment {si+1} (2x odds: {seg_dt:.0f}d)',
                        hovertext=hover_seg, hoverinfo='text',
                    ))
        else:
            rli_ols_params = fit_line(days_used, logit_used)
            rli_ols_dt = np.log(2) / rli_ols_params[1] if rli_ols_params[1] > 0 else float('inf')
            # Historical portion: OLS through data points
            d0 = int(days_used[0])
            d_last = int(days_used[-1])
            days_range = np.arange(d0, d_last + 1, 1)
            logit_trend = rli_ols_params[0] + rli_ols_params[1] * days_range
            y_trend = _inv_logit(logit_trend) * 100
            dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
            hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_seg, y_trend)]
            fig.add_trace(go.Scatter(
                x=dates_seg, y=y_trend.tolist(),
                mode='lines', line=dict(color='#2c3e50', width=2.5),
                name=f'OLS trend (2x odds: {rli_ols_dt:.0f}d)',
                hovertext=hover_seg, hoverinfo='text',
            ))
            # Projected portion: user DT slope from last data point
            _user_dt_center = np.sqrt(rli_custom_dt_lo * rli_custom_dt_hi)
            _user_logit_slope = np.log(2) / _user_dt_center
            _ols_logit_at_last = rli_ols_params[0] + rli_ols_params[1] * d_last
            _proj_intercept = _ols_logit_at_last - _user_logit_slope * d_last
            d1 = (proj_end_date - base_date).days
            days_proj = np.arange(d_last, d1 + 1, 1)
            logit_proj = _proj_intercept + _user_logit_slope * days_proj
            y_proj = _inv_logit(logit_proj) * 100
            dates_proj = [base_date + timedelta(days=int(d)) for d in days_proj]
            hover_proj = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_proj, y_proj)]
            fig.add_trace(go.Scatter(
                x=dates_proj, y=y_proj.tolist(),
                mode='lines', line=dict(color='#2980b9', width=2.5),
                name=f'Projection (2x odds: {_user_dt_center:.0f}d, CI {rli_custom_dt_lo}\u2013{rli_custom_dt_hi}d)',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif rli_proj_basis == "Superexponential (logit)":
        # Historical portion: fit curve through data
        d_start = int(days_used[0])
        d_last = int(days_used[-1])
        days_hist = np.arange(d_start, d_last + 1, 1)
        logit_hist = _rli_se_A + _rli_se_K * 2 ** (days_hist / rli_superexp_halflife)
        y_hist = _inv_logit(logit_hist) * 100
        dates_hist = [base_date + timedelta(days=int(d)) for d in days_hist]
        hover_hist = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_hist, y_hist)]
        fig.add_trace(go.Scatter(
            x=dates_hist, y=y_hist.tolist(),
            mode='lines', line=dict(color='#2c3e50', width=2.5),
            name=f'Superexp fit (2x odds: {rli_superexp_dt_fitted:.0f}d, HL={rli_superexp_halflife}d)',
            hovertext=hover_hist, hoverinfo='text',
        ))
        # Projected portion: use same formula as trajectories with center DT
        _rli_user_dt = np.sqrt(rli_superexp_dt_ci_lo * rli_superexp_dt_ci_hi)
        d_end = (proj_end_date - base_date).days
        days_proj = np.arange(0, d_end - d_last + 1, 1)
        logit_proj_growth = np.log(2) * superexp_trajectory(
            days_proj, _rli_user_dt, rli_superexp_halflife, rli_superexp_dt_floor)
        logit_proj = _rli_se_fitted_logit + logit_proj_growth
        y_proj = _inv_logit(logit_proj) * 100
        dates_proj = [rli_current['date'] + timedelta(days=int(d)) for d in days_proj]
        hover_proj = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_proj, y_proj)]
        fig.add_trace(go.Scatter(
            x=dates_proj, y=y_proj.tolist(),
            mode='lines', line=dict(color='#2980b9', width=2.5),
            name=f'Projection (2x odds: {_rli_user_dt:.0f}d, CI {rli_superexp_dt_ci_lo}\u2013{rli_superexp_dt_ci_hi}d)',
            hovertext=hover_proj, hoverinfo='text',
        ))

    # --- Milestone hlines ---
    if rli_show_milestones:
        x_lo = rli_all[0]['date'] - timedelta(days=30)
        x_hi = proj_end_date
        for score_val, label, color in [
            (5,  "RLI 5%",  '#888888'),
            (10, "RLI 10%", '#666666'),
            (25, "RLI 25%", '#c0392b'),
            (50, "RLI 50%", '#8e44ad'),
        ]:
            fig.add_trace(go.Scatter(
                x=[x_lo, x_hi], y=[score_val, score_val],
                mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=1.0, xref='paper', y=score_val, text=f"  {label}",
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=color))


    today = _add_today_vline(fig)

    # --- Backtesting ---
    rli_is_backtesting = rli_proj_as_of_idx < len(rli_frontier_all) - 1
    rli_backtest_results = []
    _rli_bt_lookup = {}
    if rli_is_backtesting:
        _rli_bt_future = rli_frontier_all[rli_proj_as_of_idx + 1:]
        rli_backtest_results = _backtest_stats(
            _rli_bt_future, all_trajectories, rli_current['date'], proj_end_date,
            lambda m: m['rli_score'],
            lambda m: m['name'],
        )
        _rli_bt_lookup = {r['name']: r for r in rli_backtest_results}

    # --- Data points ---
    for m in rli_all:
        if m['is_frontier']:
            continue
        hover = f"{m['name']}<br>{m['date'].strftime('%b %d, %Y')}<br>RLI: {m['rli_score']:.2f}%"
        fig.add_trace(go.Scatter(
            x=[m['date']], y=[m['rli_score']],
            mode='markers' + ('+text' if rli_show_labels else ''),
            marker=dict(color='#aaaaaa', size=6, symbol='circle-open',
                        line=dict(color='#bbbbbb', width=1)),
            text=[m['name']] if rli_show_labels else None,
            textposition='top right',
            textfont=dict(size=8, color='#bbbbbb'),
            hovertext=hover, hoverinfo='text', showlegend=False,
        ))

    for idx_m, m in enumerate(rli_frontier_all):
        is_used = idx_m <= rli_proj_as_of_idx
        is_selected = idx_m == rli_proj_as_of_idx
        hover = f"{m['name']}<br>{m['date'].strftime('%b %d, %Y')}<br>RLI: {m['rli_score']:.2f}%"

        if is_used:
            color = '#e74c3c' if is_selected else '#4F8DFD'
            sym = 'star' if is_selected else 'circle'
            sz = 14 if is_selected else 10
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['rli_score']],
                mode='markers' + ('+text' if rli_show_labels else ''),
                marker=dict(color=color, size=sz, symbol=sym,
                            line=dict(color='white', width=1)),
                text=[m['name']] if rli_show_labels else None,
                textposition='top right',
                textfont=dict(size=9, color='#c0392b' if is_selected else '#1a1a2e'),
                hovertext=hover, hoverinfo='text', showlegend=False,
            ))
        else:
            _rli_bt_name = m['name']
            if rli_is_backtesting and _rli_bt_name in _rli_bt_lookup:
                r = _rli_bt_lookup[_rli_bt_name]
                _btc = _bt_color_for(r)
                _bt_label = f"{_rli_bt_name} (p{r['percentile']:.0f})"
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[m['rli_score']],
                    mode='markers+text',
                    marker=dict(color=_btc, size=12, symbol='diamond',
                                line=dict(color='white', width=1)),
                    text=[_bt_label],
                    textposition='top right',
                    textfont=dict(size=9, color=_btc),
                    hovertext=hover + f"<br>Percentile: {r['percentile']:.0f}%",
                    hoverinfo='text', showlegend=False,
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[m['date']], y=[m['rli_score']],
                    mode='markers' + ('+text' if rli_show_labels else ''),
                    marker=dict(color='#aaaaaa', size=10, symbol='circle-open',
                                line=dict(color='#777777', width=2)),
                    text=[m['name']] if rli_show_labels else None,
                    textposition='top right',
                    textfont=dict(size=9, color='#999999'),
                    hovertext=hover, hoverinfo='text', showlegend=False,
                ))

    # --- Backtest overlay ---
    if rli_is_backtesting and rli_backtest_results:
        _add_backtest_traces(fig, rli_backtest_results, rli_current['date'])

    # --- Layout ---
    if rli_use_log_scale:
        _rli_y_min_data = min(m['rli_score'] for m in rli_all)
        y_min = _rli_y_min_data * 0.5
        y_max = min(max(pct95[-1], max(m['rli_score'] for m in rli_all) + 2, 55) + 5, 105)
        yaxis_cfg = dict(
            title="RLI Score (%, log scale)",
            type='log',
            range=[np.log10(y_min), np.log10(y_max)],
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            ticksuffix='%',
            tickfont=dict(color='#1a1a2e'),
            title_font=dict(color='#1a1a2e'),
        )
    else:
        y_max = min(max(pct95[-1], max(m['rli_score'] for m in rli_all) + 2, 55) + 5, 105)
        yaxis_cfg = dict(
            title="RLI Score (%)",
            range=[0, y_max],
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            ticksuffix='%',
            tickfont=dict(color='#1a1a2e'),
            title_font=dict(color='#1a1a2e'),
        )

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=140, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(
            range=[rli_all[0]['date'] - timedelta(days=30),
                   proj_end_date + timedelta(days=30)],
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(color='#1a1a2e'),
            zeroline=False,
        ),
        yaxis=yaxis_cfg,
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)',
                    font=dict(color='#1a1a2e')),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # ── Render chart + metrics ──────────────────────────────────────────────
    st.plotly_chart(fig, width="stretch")
    if rli_is_backtesting and rli_backtest_results:
        _backtest_summary(rli_backtest_results)

    # ── Projections row ───────────────────────────────────────────────────
    rli_start_logit = rli_proj_start_logit
    rli_current_label = rli_current['name']

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026EOY", datetime(2026, 12, 31)),
        ("2027 Jun EOM", datetime(2027, 6, 30)),
        ("2027EOY", datetime(2027, 12, 31)),
        ("2028EOY", datetime(2028, 12, 31)),
        ("2029EOY", datetime(2029, 12, 31)),
    ]

    def _proj_rli_at(elapsed_days, start_logit, logit_slopes, superexp=False, hl=None, slope_floor_val=None):
        """Project RLI score forward by elapsed_days. Returns percentage (0-100)."""
        if superexp and hl is not None:
            if slope_floor_val is not None and slope_floor_val > 0:
                t_cap = np.where(logit_slopes < slope_floor_val, hl * np.log2(slope_floor_val / logit_slopes), 0.0)
                se_phase = np.minimum(elapsed_days, t_cap)
                logit_se = (hl / np.log(2)) * logit_slopes * (2**(se_phase / hl) - 1)
                logit_lin = np.maximum(elapsed_days - t_cap, 0) * slope_floor_val
                logit_total = logit_se + logit_lin
            else:
                logit_total = (hl / np.log(2)) * logit_slopes * (2**(elapsed_days / hl) - 1)
        else:
            logit_total = elapsed_days * logit_slopes
        return _inv_logit(start_logit + logit_total) * 100

    _rli_slope_floor = np.log(2) / rli_superexp_dt_floor if rli_is_superexp else None

    all_targets = [
        (f"{rli_current_label} ({rli_current['date'].strftime('%b %Y')})", rli_current['date']),
    ] + eoy_targets
    n_all_cols = len(all_targets)
    cols = st.columns([1.2] + [1] * (n_all_cols - 1))
    for col, (label, target_date) in zip(cols, all_targets):
        elapsed = (target_date - rli_current['date']).days
        proj_scores = _proj_rli_at(
            elapsed, rli_start_logit, rli_proj_logit_slope,
            rli_is_superexp, rli_superexp_halflife, _rli_slope_floor)
        p10_s, p50_s, p90_s = np.percentile(proj_scores, [10, 50, 90])
        display_s = rli_current_score if elapsed == 0 else p50_s
        with col:
            st.metric(label=label, value=f"{display_s:.1f}%")
            st.caption(f"80% CI: {p10_s:.1f}% \u2013 {p90_s:.1f}%")

    # Milestone tables
    rli_milestone_thresholds = [
        (5,  "RLI 5%"),
        (10, "RLI 10%"),
        (25, "RLI 25%"),
        (50, "RLI 50%"),
    ]

    with st.expander("Milestone details"):
        tcol1, tcol2 = st.columns(2)

        with tcol1:
            st.markdown("**Probabilities**")
            rows = []
            for score_threshold, ms_label in rli_milestone_thresholds:
                row = {"Milestone": ms_label}
                for eoy_label, target_date in eoy_targets:
                    elapsed = (target_date - rli_current['date']).days
                    proj_scores = _proj_rli_at(
                        elapsed, rli_start_logit, rli_proj_logit_slope,
                        rli_is_superexp, rli_superexp_halflife, _rli_slope_floor)
                    prob = np.mean(proj_scores >= score_threshold) * 100
                    row[eoy_label] = f"{prob:.0f}%"
                rows.append(row)
            st.table(rows)

        with tcol2:
            st.markdown("**Estimated arrival**")
            arrival_rows = []
            # For arrival estimates, simulate forward in time
            for score_threshold, ms_label in rli_milestone_thresholds:
                logit_threshold = _logit(score_threshold / 100)
                logit_needed = logit_threshold - rli_start_logit
                if rli_is_superexp and rli_superexp_halflife is not None:
                    slope_fl = np.log(2) / rli_superexp_dt_floor
                    t_cap = np.where(rli_proj_logit_slope < slope_fl,
                                     rli_superexp_halflife * np.log2(slope_fl / rli_proj_logit_slope), 0.0)
                    logit_at_cap = (rli_superexp_halflife / np.log(2)) * rli_proj_logit_slope * (2**(t_cap / rli_superexp_halflife) - 1)
                    arg = 1 + logit_needed * np.log(2) / (rli_proj_logit_slope * rli_superexp_halflife)
                    arg = np.maximum(arg, 1e-10)
                    days_se_only = rli_superexp_halflife * np.log2(arg)
                    leftover = np.maximum(logit_needed - logit_at_cap, 0)
                    days_with_floor = t_cap + leftover / slope_fl
                    days_to = np.where(logit_needed <= logit_at_cap, days_se_only, days_with_floor)
                else:
                    days_to = logit_needed / rli_proj_logit_slope
                days_to = np.maximum(days_to, 0)
                p10_d, p50_d, p90_d = np.percentile(days_to, [10, 50, 90])
                med_date = rli_current['date'] + timedelta(days=max(p50_d, 0))
                early_date = rli_current['date'] + timedelta(days=max(p10_d, 0))
                late_date = rli_current['date'] + timedelta(days=max(p90_d, 0))
                arrival_rows.append({
                    "Milestone": ms_label,
                    "Median": med_date.strftime('%b %Y'),
                    "80% CI": f"{early_date.strftime('%b %Y')} \u2013 {late_date.strftime('%b %Y')}",
                })
            st.table(arrival_rows)

    _fn_caption(
        "RLI = Remote Labor Index (remotelabor.ai), fitted in logit space. "
        "These are projections, not forecasts.",
        ("fitted in logit space", "Which keeps projected scores bounded "
                                  "0\u2013100%."),
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))


# ── RSI (CoBench: automating Anthropic's own AI R&D) ─────────────────────

_RSI_RESET_KEYS = [
    "rsi_end_year", "rsi_timing", "rsi_notyet", "rsi_notyet_ramp",
    "rsi_custom_dt_lo", "rsi_custom_dt_hi", "rsi_custom_dt_dist",
    "rsi_custom_pos_lo", "rsi_custom_pos_hi",
]

_RSI_DEFAULTS = {
    "rsi_end_year": 2028,
    "rsi_timing": "Training run finished",
    "rsi_notyet": True,
    "rsi_notyet_ramp": 90.0,
    "rsi_custom_dt_dist": "Lognormal",
}


def _rsi_fit(frontier):
    """OLS through the frontier in logit space.

    CoBench is a bounded success rate, so the trend is fitted on the log-odds
    the way RLI and UK Cyber are — a score-space line runs through 100%.
    Returns (base_date, intercept, slope_per_day).
    """
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    logit = _logit(np.array([m['cobench'] / 100 for m in frontier]))
    if len(frontier) < 2:
        return base, logit[0], 0.0
    params = fit_line(days, logit)
    return base, params[0], params[1]


# 80% two-sided Student-t multipliers by residual dof, for rate CIs fitted on
# very few points; 1.282 is the normal limit the table decays to.
_DT_T80 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476}
# Slow edge when a tiny-sample t-interval cannot exclude a flat slope:
# ~4-year doubling, effectively flat at every horizon the app offers.
_DT_CAP_DAYS = 1461.0


def _dt_t_interval(days, ys, intercept, slope):
    """80% t-interval on an OLS doubling time (days), or None under 3 points.

    With n points the slope has n-2 residual dof; at n=3 the multiplier is
    3.08, so the interval says what a three-point fit honestly can: not much.
    It tightens automatically as points accumulate. A slow bound past
    `_DT_CAP_DAYS` — including a slope CI reaching zero — returns the cap.
    """
    days = np.asarray(days, dtype=float)
    ys = np.asarray(ys, dtype=float)
    dof = len(days) - 2
    if dof < 1 or slope <= 0:
        return None
    resid = ys - intercept - slope * days
    sxx = float(((days - days.mean()) ** 2).sum())
    if sxx <= 0:
        return None
    se = np.sqrt(float(resid @ resid) / dof / sxx)
    t = _DT_T80.get(dof, 1.282)
    lo = np.log(2) / (slope + t * se)
    s_lo = slope - t * se
    hi = (_DT_CAP_DAYS if s_lo <= np.log(2) / _DT_CAP_DAYS
          else np.log(2) / s_lo)
    return float(lo), float(hi)


def _rsi_dt_ci(frontier, fit_dt):
    """Default 80% CI on the odds-doubling time, in days.

    Every other tab defaults to the fitted rate halved and doubled, which
    assumes enough points for the fit to mean something. Three frontier points
    whose two segments disagree by ~8x do not, so the consecutive-segment rates
    widen that interval wherever they fall outside it, and the slope's own 80%
    t-interval (`_dt_t_interval` — one residual dof today, so it reaches the
    flat-slope cap) widens it further — never narrow it.
    """
    lo, hi = max(5.0, fit_dt / 2), fit_dt * 2
    for a, b in zip(frontier, frontier[1:]):
        d = (b['date'] - a['date']).days
        gain = _logit(b['cobench'] / 100) - _logit(a['cobench'] / 100)
        if d <= 0 or gain <= 0:
            continue
        seg_dt = np.log(2) * d / gain
        lo, hi = min(lo, seg_dt), max(hi, seg_dt)
    base, icpt, slope = _rsi_fit(frontier)
    tband = _dt_t_interval(
        [(m['date'] - base).days for m in frontier],
        _logit(np.array([m['cobench'] / 100 for m in frontier])), icpt, slope)
    if tband:
        lo, hi = min(lo, tband[0]), max(hi, tband[1])
    return float(round(max(5.0, lo))), float(round(hi))


def _dt_ci_t_widened(days, ys, fit_dt):
    """[DT/2, DT*2] widened — never narrowed — to the slope's 80% t-interval.

    The convention every other tab uses, made honest about small samples: with
    few points `_dt_t_interval` is wide and dominates, and it tightens back to
    the convention as points accumulate.
    """
    days = np.asarray(days, dtype=float)
    ys = np.asarray(ys, dtype=float)
    icpt, slope = fit_line(days, ys)
    lo, hi = max(5.0, round(fit_dt / 2)), round(fit_dt * 2)
    tband = _dt_t_interval(days, ys, icpt, slope)
    if tband:
        lo, hi = min(lo, tband[0]), max(hi, tband[1])
    return float(round(max(5.0, lo))), float(round(hi))


def _rsi_survey_dt_ci(rows, fit_dt):
    """Default 80% CI on the survey's doubling time, in days.

    `_dt_ci_t_widened` over every point the fit uses (the carried-over
    `estimated` round included, like the fit itself).
    """
    return _dt_ci_t_widened(
        [(r['date'] - rows[0]['date']).days for r in rows],
        np.log([r['uplift'] for r in rows]), fit_dt)


def _rsi_dir_fit(frontier):
    """OLS through the research-direction frontier, in logit space.

    A win rate is bounded, so it is fitted on the log-odds the way CoBench and
    RLI are. Returns (base_date, intercept, slope_per_day).
    """
    base = frontier[0]['date']
    days = np.array([(r['date'] - base).days for r in frontier], dtype=float)
    ys = _logit(np.array([r['better'] for r in frontier]) / 100)
    if len(frontier) < 2:
        return base, float(ys[0]), 0.0
    icpt, slope = fit_line(days, ys)
    return base, icpt, slope


def _rsi_dir_dt_ci(frontier, fit_dt):
    """Default 80% CI on the odds-doubling time, in days."""
    return _dt_ci_t_widened(
        [(r['date'] - frontier[0]['date']).days for r in frontier],
        _logit(np.array([r['better'] for r in frontier]) / 100), fit_dt)


def _rsi_code_fit(rows):
    """OLS on log(multiple) over the fitted bars. (base_date, icpt, slope).

    An output multiple has no ceiling, so it compounds the way the staff
    survey does rather than saturating like a percentage. Only the bars from
    `_RSI_CODE_FIT_FROM` are fitted — the earlier ones are the baseline the
    multiple is taken against.
    """
    fr = [r for r in rows if r['fitted']]
    base = fr[0]['date']
    days = np.array([(r['date'] - base).days for r in fr], dtype=float)
    logs = np.log(np.array([r['mult'] for r in fr]))
    if len(fr) < 2:
        return base, float(logs[0]), 0.0
    icpt, slope = fit_line(days, logs)
    return base, icpt, slope


def _rsi_code_dt_ci(rows, fit_dt):
    """Default 80% CI on the doubling time, in days."""
    fr = [r for r in rows if r['fitted']]
    return _dt_ci_t_widened([(r['date'] - fr[0]['date']).days for r in fr],
                            np.log([r['mult'] for r in fr]), fit_dt)


def render_rsi():
    if st.session_state.pop("_reset_rsi", False):
        for k in _RSI_RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(_RSI_DEFAULTS)
        st.rerun()

    for k, v in _RSI_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    _base, _intercept, _slope = _rsi_fit(rsi_frontier_all)
    _fit_dt = round(np.log(2) / _slope) if _slope > 0 else 200
    current = rsi_frontier_all[-1]

    with st.sidebar:
        st.header("RSI Projection")
        rsi_end_year = st.selectbox("Project through", [2027, 2028, 2029, 2030, 2031],
                                    key="rsi_end_year")
        if st.session_state.get("rsi_timing") not in _DC_TIMING_OPTIONS:
            st.session_state["rsi_timing"] = _RSI_DEFAULTS["rsi_timing"]
        rsi_timing = st.selectbox(
            "Milestone dates point at", list(_DC_TIMING_OPTIONS), key="rsi_timing",
            help="Which moment the Capabilities Milestone dates mean. Benchmark "
                 "milestones are dated off released models, so anything earlier "
                 "than release pulls them back — see the note under the cards.")
        with st.expander("Advanced options"):
            st.button("Reset to defaults", key="reset_rsi",
                      on_click=lambda: st.session_state.update(_reset_rsi=True))
            _default_dt_lo, _default_dt_hi = _rsi_dt_ci(rsi_frontier_all, _fit_dt)
            _dt_lo_col, _dt_hi_col = st.columns(2)
            rsi_dt_lo = _ss_number_input(
                _dt_lo_col, "Odds 2x time CI low (days)", "rsi_custom_dt_lo",
                _default_dt_lo, min_value=5.0, max_value=2000.0,
                step=5.0, help="Fast scenario: days for the odds p/(1-p) to double.")
            rsi_dt_hi = _ss_number_input(
                _dt_hi_col, "Odds 2x time CI high (days)", "rsi_custom_dt_hi",
                _default_dt_hi, min_value=5.0, max_value=5000.0, step=5.0,
                help="Slow scenario. The default 80% CI is the fitted rate halved "
                     "and doubled, widened to span the consecutive-segment rates.")
            rsi_dt_dist = st.selectbox("Rate distribution", ["Lognormal", "Normal"],
                                       key="rsi_custom_dt_dist")
            _pos_lo_col, _pos_hi_col = st.columns(2)
            rsi_pos_lo = _ss_number_input(
                _pos_lo_col, "Start CI low (%)", "rsi_custom_pos_lo",
                float(round(max(1.0, current['cobench'] - 10), 1)),
                min_value=0.5, max_value=99.0, step=1.0,
                help="80% CI on where the frontier actually stands today.")
            rsi_pos_hi = _ss_number_input(
                _pos_hi_col, "Start CI high (%)", "rsi_custom_pos_hi",
                float(round(min(99.0, current['cobench'] + 10), 1)),
                min_value=0.5, max_value=99.5, step=1.0)

    st.header("RSI")
    st.warning("Warning: RSI projections under construction, these are simple "
               "models and not 'all things considered' takes.")

    st.subheader("CoBench")
    _fn_line(
        "Recursive self-improvement runs through a lab automating its own "
        "research. **CoBench** is Anthropic's internal measure of that.",
        ("internal measure of that", "A model is dropped into a historical "
                                     "snapshot of Anthropic's codebase, logs, "
                                     "internal messaging and docs, and asked to "
                                     "diagnose the root cause of an issue "
                                     "Anthropic engineers actually solved. 449 "
                                     "problems, model-graded against the root "
                                     "cause found in practice."))

    # ── Trajectories ─────────────────────────────────────────────────────
    if rsi_dt_lo > rsi_dt_hi:
        rsi_dt_lo, rsi_dt_hi = rsi_dt_hi, rsi_dt_lo
    if rsi_pos_lo > rsi_pos_hi:
        rsi_pos_lo, rsi_pos_hi = rsi_pos_hi, rsi_pos_lo

    n = N_SAMPLES
    proj_dt = (_lognormal_from_ci(rsi_dt_lo, rsi_dt_hi, n) if rsi_dt_dist == "Lognormal"
               else _normal_from_ci(rsi_dt_lo, rsi_dt_hi, n))
    proj_dt = np.maximum(proj_dt, 1.0)
    proj_slope = np.log(2) / proj_dt

    _pos_sigma = (_logit(rsi_pos_hi / 100) - _logit(rsi_pos_lo / 100)) / (2 * 1.282)
    _fitted_logit = _intercept + _slope * (current['date'] - _base).days
    proj_start_logit = np.random.normal(_fitted_logit, max(_pos_sigma, 0), n)

    proj_end_date = datetime(rsi_end_year, 12, 31)
    proj_days = np.arange(0, (proj_end_date - current['date']).days + 1, dtype=float)
    proj_dates = [current['date'] + timedelta(days=int(d)) for d in proj_days]
    traj = _inv_logit(proj_start_logit[:, None] + proj_days[None, :] * proj_slope[:, None]) * 100

    pct = {q: np.percentile(traj, q, axis=0) for q in (5, 10, 25, 50, 75, 90, 95)}

    fig = go.Figure()
    for lo, hi, color, label in [(5, 95, 'rgba(52,152,219,0.10)', '90% CI'),
                                 (10, 90, 'rgba(52,152,219,0.18)', '80% CI'),
                                 (25, 75, 'rgba(52,152,219,0.28)', '50% CI')]:
        fig.add_trace(go.Scatter(
            x=proj_dates + proj_dates[::-1],
            y=list(pct[hi]) + list(pct[lo][::-1]),
            fill='toself', fillcolor=color, line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True))

    # Fitted trend through the recorded frontier, then the median projection.
    _hist_days = np.arange(0, (current['date'] - _base).days + 1, dtype=float)
    _hist_y = _inv_logit(_intercept + _slope * _hist_days) * 100
    _hist_dates = [_base + timedelta(days=int(d)) for d in _hist_days]
    fig.add_trace(go.Scatter(
        x=_hist_dates, y=_hist_y.tolist(), mode='lines',
        line=dict(color='#2c3e50', width=2.5),
        name=f"Fitted trend (2x odds: {_fit_dt:.0f}d)",
        hovertext=[f"{d.strftime('%b %d, %Y')}<br>Trend: {y:.1f}%"
                   for d, y in zip(_hist_dates, _hist_y)],
        hoverinfo='text'))
    fig.add_trace(go.Scatter(
        x=proj_dates, y=pct[50].tolist(), mode='lines',
        line=dict(color='#2c3e50', width=2.5, dash='dash'),
        name='Median projection',
        hovertext=[f"{d.strftime('%b %d, %Y')}<br>Median: {y:.1f}%"
                   for d, y in zip(proj_dates, pct[50])],
        hoverinfo='text'))

    for m in rsi_all:
        _is_fr = m['is_frontier']
        fig.add_trace(go.Scatter(
            x=[m['date']], y=[m['cobench']],
            mode='markers+text',
            marker=dict(color='#4F8DFD' if _is_fr else '#aaaaaa', size=12,
                        symbol='circle' if _is_fr else 'circle-open',
                        line=dict(color='white' if _is_fr else '#777777', width=2)),
            text=[m['name']],
            textposition='top center',
            textfont=dict(size=10, color='#1a1a2e' if _is_fr else '#999999'),
            hovertext=f"{m['name']}<br>{_rsi_date_label(m)}<br>"
                      f"CoBench: {m['cobench']:.1f}%",
            hoverinfo='text', showlegend=False))

    fig.add_hline(
        y=_RSI_SUBSTITUTION_BAR, line=dict(color='#e74c3c', width=1.5, dash='dash'),
        annotation_text=f"{_RSI_SUBSTITUTION_BAR:.0f}% — full substitution for "
                        "Anthropic research staff",
        annotation_position="top left",
        annotation_font=dict(size=11, color='#e74c3c'))
    _add_today_vline(fig)

    fig.update_layout(
        height=600,
        margin=dict(l=50, r=60, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(title="Release date",
                   range=[rsi_all[0]['date'] - timedelta(days=25),
                          proj_end_date + timedelta(days=25)],
                   gridcolor='rgba(0,0,0,0.1)',
                   tickfont=dict(color='#1a1a2e'), zeroline=False,
                   title_font=dict(color='#1a1a2e')),
        yaxis=dict(title="CoBench score (%)", range=[0, 100],
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False, ticksuffix='%',
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        legend=dict(yanchor='bottom', y=0.02, xanchor='right', x=0.98,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Source: "
        f"[Anthropic, Redacted Risk Report, August 2026, §3.4.3]({_RSI_SOURCE_URL}); "
        "scores read off Figure 3.4.3.A.")

    _render_rsi_survey()
    _render_rsi_code()
    _render_rsi_direction(rsi_end_year)

    st.markdown("---")
    # The conditioning controls render inside the blend's weights expander,
    # below their consumers — read via session state, like the weights.
    _pc_render_milestones(rsi_timing, datetime.now(),
                          condition=st.session_state.get("rsi_notyet", True),
                          ramp_days=st.session_state.get(
                              "rsi_notyet_ramp",
                              _RSI_DEFAULTS["rsi_notyet_ramp"]),
                          end_year=rsi_end_year)


def _render_rsi_survey():
    """The report's other substitution series: self-reported researcher uplift.

    Fitted and projected the same way the CoBench half is, but on log(multiple)
    rather than log-odds: an output multiple has no ceiling to bound it against,
    so it compounds the way METR's horizon does.
    """
    st.subheader("Anthropic internal staff survey on speedup")

    rows = load_rsi_survey()
    base = rows[0]['date']
    days = np.array([(r['date'] - base).days for r in rows], dtype=float)
    logs = np.log(np.array([r['uplift'] for r in rows]))
    intercept, slope = fit_line(days, logs)
    cur = rows[-1]

    fig = go.Figure()

    if slope > 0:
        dt = np.log(2) / slope
        n = N_SAMPLES
        proj_slope = np.log(2) / np.maximum(
            _lognormal_from_ci(*_rsi_survey_dt_ci(rows, dt), n=n), 1.0)
        fitted = intercept + slope * (cur['date'] - base).days
        sigma = np.log(_RSI_SURVEY_POS_FACTOR) / 1.282
        start_log = np.random.normal(fitted, sigma, n)

        pdays = np.arange(0, _RSI_SURVEY_HORIZON_DAYS + 1, dtype=float)
        pdates = [cur['date'] + timedelta(days=int(d)) for d in pdays]
        traj = np.exp(start_log[:, None] + pdays[None, :] * proj_slope[:, None])
        pct = {q: np.percentile(traj, q, axis=0) for q in (5, 10, 25, 50, 75, 90, 95)}

        for lo, hi, color, label in [(5, 95, 'rgba(52,152,219,0.10)', '90% CI'),
                                     (10, 90, 'rgba(52,152,219,0.18)', '80% CI'),
                                     (25, 75, 'rgba(52,152,219,0.28)', '50% CI')]:
            fig.add_trace(go.Scatter(
                x=pdates + pdates[::-1],
                y=list(pct[hi]) + list(pct[lo][::-1]),
                fill='toself', fillcolor=color, line=dict(width=0),
                name=label, hoverinfo='skip', showlegend=True))

        hdays = np.arange(0, (cur['date'] - base).days + 1, dtype=float)
        hdates = [base + timedelta(days=int(d)) for d in hdays]
        fig.add_trace(go.Scatter(
            x=hdates, y=np.exp(intercept + slope * hdays).tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5),
            name=f"Fitted trend (2x: {dt:.0f}d)",
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Trend: ~{y:.1f}x"
                       for d, y in zip(hdates, np.exp(intercept + slope * hdays))],
            hoverinfo='text'))
        fig.add_trace(go.Scatter(
            x=pdates, y=pct[50].tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5, dash='dash'),
            name='Median projection',
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Median: ~{y:.1f}x"
                       for d, y in zip(pdates, pct[50])],
            hoverinfo='text'))
        x_end = pdates[-1]
        y_top = max(float(pct[95][-1]), _PC_RSI_SURVEY_TARGET_X * 1.3)
    else:
        x_end, y_top = cur['date'], _PC_RSI_SURVEY_TARGET_X * 1.3

    for r in rows:
        if 'lo' in r:
            fig.add_trace(go.Scatter(
                x=[r['date'], r['date']], y=[r['lo'], r['hi']],
                mode='lines', line=dict(color='#4F8DFD', width=1.5),
                opacity=0.35, hoverinfo='skip', showlegend=False))
        _est = r.get('estimated', False)
        fig.add_trace(go.Scatter(
            x=[r['date']], y=[r['uplift']], mode='markers+text',
            marker=dict(color='white' if _est else '#4F8DFD', size=13,
                        line=dict(color='#4F8DFD', width=2) if _est
                        else dict(color='white', width=2)),
            text=[r['name'] + (' (est.)' if _est else '')],
            # Below the marker for the estimate: the median projection passes
            # just above it, and a top label lands on the line.
            textposition='bottom center' if _est else 'top center',
            textfont=dict(size=10, color='#999999' if _est else '#1a1a2e'),
            hovertext=f"{r['name']}<br>{r['date'].strftime('%b %Y')}<br>"
                      f"~{r['uplift']:g}x output vs no AI<br>{r['note']}",
            hoverinfo='text', showlegend=False))

    fig.add_hline(y=_PC_RSI_SURVEY_TARGET_X,
                  line=dict(color='#e74c3c', width=1.5, dash='dash'))
    # By hand, not `annotation_text`: on a log axis plotly reads an
    # annotation's y as the exponent and parks the label off the chart.
    fig.add_annotation(
        xref='x domain', x=0.01, y=np.log10(_PC_RSI_SURVEY_TARGET_X),
        yanchor='bottom', xanchor='left', showarrow=False,
        text=f"{_PC_RSI_SURVEY_TARGET_X:.0f}x — the milestone bar, about a "
             "doubling and a half past the last round",
        font=dict(size=11, color='#e74c3c'))
    _add_today_vline(fig)
    fig.update_layout(
        height=480, margin=dict(l=50, r=60, t=40, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(title="Survey round",
                   range=[base - timedelta(days=30),
                          max(x_end, rows[-1]['date']) + timedelta(days=30)],
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        yaxis=dict(title="Self-reported output vs. no AI assistance",
                   type='log', range=[0, np.log10(y_top * 1.3)],
                   # Decades only, as on the merged-code chart: the default
                   # log axis labels every minor tick and buries them.
                   dtick=1, ticksuffix='x',
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width="stretch")

    _fn_caption(
        "Anthropic surveys its own technical staff on productivity uplift per "
        "frontier model. The rounds differ, the newest point is an assumed "
        "value, and the trend is fitted on the log of the multiple. Source: "
        f"[Redacted Risk Report, August 2026, \u00a73.4.2]({_RSI_SOURCE_URL}) "
        "and the corresponding Claude system cards.",
        ("The rounds differ", "In who was sampled and which statistic was "
                              "reported \u2014 hover a point for its own note."),
        ("an assumed value", "Mythos Preview is the most recent surveyed round: "
                             "no new survey was run for Mythos 5 or for Model 2 "
                             "(internal), which is shown at the ~4x Mythos "
                             "Preview reported. It is included in the fit, which "
                             "it flattens."),
        ("fitted on the log of the multiple", "Those four points, projected the "
                                              "way the CoBench fan above is, but "
                                              "only a year out \u2014 compounded "
                                              "further it leaves the chart, which "
                                              "is a fact about the fit rather "
                                              "than about the future."))


def _render_rsi_code():
    """Anthropic's own merged-code volume per engineer.

    The counted counterpart to the self-reported survey above, fitted the same
    way — OLS on log(multiple) — over the quarters from 2025 on.
    """
    st.subheader("Code merged per Anthropic engineer")
    _fn_line(
        "The survey asks staff how much faster they are; this counts what "
        "they merged. Lines of code per active contributor each quarter, "
        "against the pre-2025 average.",
        ("Lines of code per active contributor",
         "Per-PR line counts capped at the 99th percentile; an active "
         "contributor is a distinct author in the trailing twelve months. "
         "The models driving the recent quarters include ones Anthropic had "
         "internal access to before release, which the figure annotates."))

    rows = load_rsi_code()
    base, icpt, slope = _rsi_code_fit(rows)
    cur = rows[-1]

    fig = go.Figure()

    if slope > 0:
        dt = np.log(2) / slope
        n = N_SAMPLES
        proj_slope = np.log(2) / np.maximum(
            _lognormal_from_ci(*_rsi_code_dt_ci(rows, dt), n=n), 1.0)
        fitted = icpt + slope * (cur['date'] - base).days
        start_log = np.random.normal(
            fitted, np.log(_RSI_CODE_POS_FACTOR) / 1.282, n)

        pdays = np.arange(0, _RSI_CODE_HORIZON_DAYS + 1, dtype=float)
        pdates = [cur['date'] + timedelta(days=int(d)) for d in pdays]
        traj = np.exp(start_log[:, None] + pdays[None, :] * proj_slope[:, None])
        pct = {q: np.percentile(traj, q, axis=0) for q in (5, 10, 25, 50, 75, 90, 95)}

        for lo, hi, color, label in [(5, 95, 'rgba(52,152,219,0.10)', '90% CI'),
                                     (10, 90, 'rgba(52,152,219,0.18)', '80% CI'),
                                     (25, 75, 'rgba(52,152,219,0.28)', '50% CI')]:
            fig.add_trace(go.Scatter(
                x=pdates + pdates[::-1],
                y=list(pct[hi]) + list(pct[lo][::-1]),
                fill='toself', fillcolor=color, line=dict(width=0),
                name=label, hoverinfo='skip', showlegend=True))

        hdays = np.arange(0, (cur['date'] - base).days + 1, dtype=float)
        hdates = [base + timedelta(days=int(d)) for d in hdays]
        hy = np.exp(icpt + slope * hdays)
        fig.add_trace(go.Scatter(
            x=hdates, y=hy.tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5),
            name=f"Fitted trend (2x: {dt:.0f}d)",
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Trend: {y:.1f}x"
                       for d, y in zip(hdates, hy)],
            hoverinfo='text'))
        fig.add_trace(go.Scatter(
            x=pdates, y=pct[50].tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5, dash='dash'),
            name='Median projection',
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Median: {y:.1f}x"
                       for d, y in zip(pdates, pct[50])],
            hoverinfo='text'))
        x_end = pdates[-1]
        y_top = max(float(pct[95][-1]), _RSI_CODE_TARGET * 1.3)
    else:
        x_end, y_top = cur['date'], _RSI_CODE_TARGET * 1.3

    # The quarters themselves: a light connector so 21 points read as a
    # series, then markers — hollow grey for the unfitted baseline bars,
    # hollow blue for the partial quarter, as the survey draws its estimate.
    fig.add_trace(go.Scatter(
        x=[r['date'] for r in rows], y=[r['mult'] for r in rows],
        mode='lines', line=dict(color='rgba(79,141,253,0.30)', width=1.5),
        hoverinfo='skip', showlegend=False))
    for grp, marker in [
            ([r for r in rows if not r['fitted']],
             dict(color='#aaaaaa', size=8, symbol='circle-open',
                  line=dict(color='#777777', width=2))),
            ([r for r in rows if r['fitted'] and not r['partial']],
             dict(color='#4F8DFD', size=11,
                  line=dict(color='white', width=2))),
            ([r for r in rows if r['partial']],
             dict(color='white', size=11,
                  line=dict(color='#4F8DFD', width=2)))]:
        if not grp:
            continue
        fig.add_trace(go.Scatter(
            x=[r['date'] for r in grp], y=[r['mult'] for r in grp],
            mode='markers', marker=marker,
            hovertext=[_rsi_code_hover(r) for r in grp],
            hoverinfo='text', showlegend=False))

    fig.add_hline(y=_RSI_CODE_TARGET,
                  line=dict(color='#e74c3c', width=1.5, dash='dash'))
    # The label is added by hand rather than through `annotation_text`: on a
    # log axis plotly reads an annotation's y as the exponent, which parks
    # `annotation_text` at 10^20 and off the chart.
    fig.add_annotation(
        xref='x domain', x=0.01, y=np.log10(_RSI_CODE_TARGET),
        yanchor='bottom', xanchor='left', showarrow=False,
        text=f"{_RSI_CODE_TARGET:.0f}x — one engineer merging what "
             f"{_RSI_CODE_TARGET:.0f} did before 2025",
        font=dict(size=11, color='#e74c3c'))
    _add_today_vline(fig)

    fig.update_layout(
        height=480, margin=dict(l=50, r=60, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(title="Quarter",
                   range=[_RSI_CODE_CHART_FROM,
                          max(x_end, cur['date']) + timedelta(days=30)],
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        yaxis=dict(title="Lines merged per active contributor "
                         "(× pre-2025 average)",
                   type='log', range=[np.log10(0.5), np.log10(y_top * 1.3)],
                   # Decades only: the default log axis labels every minor
                   # tick, and three decades of them bury the four that matter.
                   dtick=1, ticksuffix='x',
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        # Top left, above the target label: the series climbs left to right,
        # so the fan owns the bottom right corner too by the end.
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width="stretch")

    _fn_caption(
        "Output volume, not research progress; the last quarter is partial "
        "and the fit uses the quarters from 2025 on. Source: "
        f"[Anthropic, *When AI builds itself*]({_RSI_CODE_SOURCE_URL}); bar "
        "values read off the figure.",
        ("Output volume", "Lines merged per contributor. A coding model "
                          "writes more lines for the same task, so some of "
                          "the multiple is verbosity rather than work done "
                          "— it measures how much code ships, not how "
                          "much research it settles."),
        ("the last quarter is partial", "The figure hatches it: it averages "
                                        "only the days observed when the "
                                        "post went up, and it is plotted at "
                                        "its quarter's midpoint like every "
                                        "other bar, which dates it later "
                                        "than the days it covers."),
        ("the quarters from 2025 on", "The earlier bars are the baseline the "
                                      "multiple is taken against — they "
                                      "average to 1 by construction, so there "
                                      "is no trend in them to fit. They are "
                                      "drawn hollow."))


def _rsi_dir_label_positions(rows, gap_days=90, rise_pts=6.0):
    """Where each point's name goes, or '' for no label.

    Nine models inside twenty-five months collide at a fixed 'top center', so
    only the frontier is labelled — the rest are hollow markers with the name
    on hover, as on the UK Cyber tab — and a frontier point landing within
    `gap_days` of the last labelled one drops below the line instead — or, if
    it sits `rise_pts` above that one, stays up and moves right, since the two
    already clear each other vertically.
    """
    pos, last_top = [], None
    for r in rows:
        if not r['is_frontier']:
            pos.append('')
        elif (last_top is not None
              and (r['date'] - last_top[0]).days < gap_days):
            if r['better'] - last_top[1] > rise_pts:
                pos.append('top right')
                last_top = (r['date'], r['better'])
            else:
                pos.append('bottom center')
                last_top = None      # below the line; the next may go up again
        else:
            pos.append('top center')
            last_top = (r['date'], r['better'])
    return pos


def _render_rsi_direction(end_year):
    """Anthropic's detour study: can the model pick a better next step?

    Fitted and projected exactly as the CoBench half is: a bounded win rate,
    so the trend runs on the log-odds, dated against the study's own 90%
    practical ceiling.
    """
    st.subheader("Research direction")
    _fn_line(
        "Automating research means choosing what to do next, not just doing "
        "it. Anthropic replayed 129 turns where one of its own researchers "
        "went the wrong way and asked each model for the next step instead.",
        ("129 turns where one of its own researchers went the wrong way",
         "Real Claude Code sessions from January–March 2026 on open-ended "
         "problems — debugging a training crash, chasing a benchmark "
         "regression. Anthropic found 129 moments where the researcher took a "
         "detour that sent the session off-track, showed each model only the "
         "work before the detour, and had a separate Claude that could see "
         "the whole session, resolution included, judge whose next step was "
         "better."))

    rows = rsi_dir_all
    frontier = rsi_dir_frontier_all
    base, intercept, slope = _rsi_dir_fit(frontier)
    fit_dt = np.log(2) / slope if slope > 0 else 0.0
    cur = frontier[-1]

    fig = go.Figure()

    if slope > 0:
        n = N_SAMPLES
        proj_slope = np.log(2) / np.maximum(
            _lognormal_from_ci(*_rsi_dir_dt_ci(frontier, round(fit_dt)), n=n), 1.0)
        fitted = intercept + slope * (cur['date'] - base).days
        sigma = (_logit((cur['better'] + _RSI_DIR_POS_CI) / 100)
                 - _logit((cur['better'] - _RSI_DIR_POS_CI) / 100)) / (2 * 1.282)
        start_logit = np.random.normal(fitted, max(float(sigma), 0), n)

        end_date = datetime(end_year, 12, 31)
        pdays = np.arange(0, max((end_date - cur['date']).days, 1) + 1, dtype=float)
        pdates = [cur['date'] + timedelta(days=int(d)) for d in pdays]
        traj = _inv_logit(start_logit[:, None]
                          + pdays[None, :] * proj_slope[:, None]) * 100
        pct = {q: np.percentile(traj, q, axis=0) for q in (5, 10, 25, 50, 75, 90, 95)}

        for lo, hi, color, label in [(5, 95, 'rgba(52,152,219,0.10)', '90% CI'),
                                     (10, 90, 'rgba(52,152,219,0.18)', '80% CI'),
                                     (25, 75, 'rgba(52,152,219,0.28)', '50% CI')]:
            fig.add_trace(go.Scatter(
                x=pdates + pdates[::-1],
                y=list(pct[hi]) + list(pct[lo][::-1]),
                fill='toself', fillcolor=color, line=dict(width=0),
                name=label, hoverinfo='skip', showlegend=True))

        hdays = np.arange(0, (cur['date'] - base).days + 1, dtype=float)
        hdates = [base + timedelta(days=int(d)) for d in hdays]
        hy = _inv_logit(intercept + slope * hdays) * 100
        fig.add_trace(go.Scatter(
            x=hdates, y=hy.tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5),
            name=f"Fitted trend (2x odds: {fit_dt:.0f}d)",
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Trend: {y:.1f}%"
                       for d, y in zip(hdates, hy)],
            hoverinfo='text'))
        fig.add_trace(go.Scatter(
            x=pdates, y=pct[50].tolist(), mode='lines',
            line=dict(color='#2c3e50', width=2.5, dash='dash'),
            name='Median projection',
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Median: {y:.1f}%"
                       for d, y in zip(pdates, pct[50])],
            hoverinfo='text'))
        x_end = pdates[-1]
    else:
        x_end = cur['date']

    for r, _pos in zip(rows, _rsi_dir_label_positions(rows)):
        _is_fr = r['is_frontier']
        fig.add_trace(go.Scatter(
            x=[r['date']], y=[r['better']],
            mode='markers+text' if _pos else 'markers',
            marker=dict(color='#4F8DFD' if _is_fr else '#aaaaaa', size=12,
                        symbol='circle' if _is_fr else 'circle-open',
                        line=dict(color='white' if _is_fr else '#777777', width=2)),
            text=[r['name'].replace('Claude ', '')] if _pos else None,
            textposition=_pos or 'top center',
            textfont=dict(size=10, color='#1a1a2e'),
            hovertext=f"{r['name']}<br>{_rsi_date_label(r)}<br>"
                      f"Better: {r['better']:.0f}% · tie: {r['tie']:.0f}%",
            hoverinfo='text', showlegend=False))

    fig.add_hline(
        y=_RSI_DIR_TARGET, line=dict(color='#e74c3c', width=1.5, dash='dash'),
        annotation_text=f"{_RSI_DIR_TARGET:.0f}% — the study's practical "
                        "ceiling: a judge shown the finished session",
        annotation_position="top left",
        annotation_font=dict(size=11, color='#e74c3c'))
    _add_today_vline(fig)

    fig.update_layout(
        height=520, margin=dict(l=50, r=60, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(title="Release date",
                   range=[rows[0]['date'] - timedelta(days=25),
                          max(x_end, rows[-1]['date']) + timedelta(days=25)],
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        yaxis=dict(title="Sampled turns where the model's next step won (%)",
                   range=[0, 100], ticksuffix='%',
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        legend=dict(yanchor='bottom', y=0.02, xanchor='right', x=0.98,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, width="stretch")

    _fn_caption(
        "The turns were chosen for having room for improvement, and the bar "
        "is the study's own ceiling. Source: "
        f"[Anthropic, *When AI builds itself*]({_RSI_DIR_SOURCE_URL}).",
        ("chosen for having room for improvement",
         "So this is not “how often Claude out-researches a human”: "
         "the sample is turns where the researcher's own next move was "
         "already known to be a detour. It measures the trend in that "
         "comparison, not its level."),
        ("the study's own ceiling",
         f"{_RSI_DIR_TARGET:.0f}% is what a model shown the complete session, "
         "resolution included, scores against the same researchers — the only "
         "bar the study names, and the one the fan is dated against. The fan "
         "is a log-odds trend, so it carries on past that line; nothing in "
         "the study says a model gets there."))


# ── UK Cyber (AISI narrow cyber tasks) ───────────────────────────────────

_UKC_RESET_KEYS = [
    "ukc_proj_basis", "ukc_custom_dt_lo", "ukc_custom_dt_hi",
    "ukc_custom_pos_lo", "ukc_custom_pos_hi", "ukc_custom_dt_dist",
    "ukc_piecewise_n_seg", "ukc_bp1_select",
    "ukc_superexp_halflife", "ukc_superexp_dt_floor",
    "ukc_superexp_dt_ci_lo", "ukc_superexp_dt_ci_hi",
    "ukc_labels", "ukc_show_open", "ukc_show_lag",
    "_ukc_proj_as_of", "ukc_end_year", "_ukc_seg_config",
]

_UKC_DEFAULTS = {
    "ukc_proj_basis": "Linear (logit)",
    "ukc_piecewise_n_seg": 1,
    "ukc_custom_dt_dist": "Lognormal",
    "ukc_labels": True,
    "ukc_show_open": True,
    "ukc_show_lag": True,
    "ukc_end_year": 2027,
}

# Open-weight models are coloured by country so the confound stays visible.
# Deliberately not blue: the closed frontier already owns #4F8DFD, and the
# open-vs-closed contrast is the whole point of the tab.
_UKC_OPEN_COLORS = {"China": "#8e44ad", "US": "#e67e22"}


def render_ukcyber():
    if st.session_state.pop("_reset_ukc", False):
        for k in _UKC_RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(_UKC_DEFAULTS)
        st.rerun()

    for k, v in _UKC_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Cyber Projection")

        ukc_as_of_name = st.session_state.get('_ukc_proj_as_of', ukc_frontier_names[-1])
        if ukc_as_of_name not in ukc_frontier_names:
            ukc_as_of_name = ukc_frontier_names[-1]
        ukc_as_of_idx = ukc_frontier_names.index(ukc_as_of_name)

        ukc_basis = st.radio(
            "Projection basis",
            ["Linear (logit)", "Piecewise linear (logit)", "Superexponential (logit)"],
            key="ukc_proj_basis",
            help="All projections fit in logit space so success rates stay bounded 0–100%.")
        ukc_is_superexp = ukc_basis == "Superexponential (logit)"

        # Pre-fit the closed frontier for data-driven CI defaults.
        _ukc_fr_used = ukc_frontier_all[:ukc_as_of_idx + 1]
        _ukc_base = ukc_frontier_all[0]['date']
        _ukc_days = np.array([(m['date'] - _ukc_base).days for m in _ukc_fr_used], dtype=float)
        _ukc_logit = _logit(np.array([m['cyber_score'] / 100 for m in _ukc_fr_used]))
        _ukc_params = fit_line(_ukc_days, _ukc_logit) if len(_ukc_fr_used) >= 2 else np.array([0.0, 0.007])
        _ukc_ols_dt = round(np.log(2) / _ukc_params[1]) if _ukc_params[1] > 0 else 100

        ukc_bps = []
        ukc_n_segments = 1
        ukc_dt_lo = ukc_dt_hi = None
        ukc_pos_lo = ukc_pos_hi = None
        ukc_dt_dist = "Lognormal"
        ukc_halflife = ukc_dt_floor = None
        ukc_se_dt_lo = ukc_se_dt_hi = None

        _ukc_names_used = [m['name'] for m in _ukc_fr_used]

        with st.expander("Advanced options"):
            st.button("Reset to defaults", key="reset_ukc",
                      on_click=lambda: st.session_state.update(_reset_ukc=True))

            if ukc_basis == "Piecewise linear (logit)":
                if st.session_state.get("ukc_piecewise_n_seg", 1) < 2:
                    st.session_state["ukc_piecewise_n_seg"] = 2
                _seg_opts = [2, 3] if len(_ukc_names_used) >= 5 else [2]
                ukc_n_segments = st.radio("Segments", _seg_opts, horizontal=True,
                                          key="ukc_piecewise_n_seg")
                if len(_ukc_names_used) >= 3:
                    _bp_choices = _ukc_names_used[1:-1] or _ukc_names_used[1:]
                    _bp1 = st.selectbox("Breakpoint", _bp_choices,
                                        index=len(_bp_choices) // 2, key="ukc_bp1_select")
                    ukc_bps.append(_bp1)
            else:
                st.session_state.pop("ukc_piecewise_n_seg", None)

            # Slope defaults come from the last segment actually being extrapolated.
            _seg_start = _ukc_names_used.index(ukc_bps[-1]) if ukc_bps and ukc_bps[-1] in _ukc_names_used else 0
            _seg_days, _seg_logit = _ukc_days[_seg_start:], _ukc_logit[_seg_start:]
            if len(_seg_days) >= 2:
                _seg_params = fit_line(_seg_days, _seg_logit)
                _seg_dt = round(np.log(2) / _seg_params[1]) if _seg_params[1] > 0 else _ukc_ols_dt
            else:
                _seg_dt = _ukc_ols_dt

            # Reset slope CIs when the segmentation changes under them.
            _seg_config = (ukc_basis, ukc_n_segments, tuple(ukc_bps))
            if st.session_state.get("_ukc_seg_config") != _seg_config:
                st.session_state["_ukc_seg_config"] = _seg_config
                st.session_state.pop("ukc_custom_dt_lo", None)
                st.session_state.pop("ukc_custom_dt_hi", None)

            _cur_score = _ukc_fr_used[-1]['cyber_score']

            if ukc_is_superexp:
                _c1, _c2 = st.columns(2)
                ukc_halflife = _ss_number_input(_c1, "Rate half-life (days)",
                    "ukc_superexp_halflife", 365, min_value=30, max_value=5000, step=30,
                    help="How quickly the rate accelerates. Lower = faster.")
                ukc_dt_floor = _ss_number_input(_c2, "Min odds 2x time (days)",
                    "ukc_superexp_dt_floor", 15.0, min_value=1.0, max_value=500.0, step=1.0,
                    help="Rate can't exceed this. Prevents runaway projections.")
                _s1, _s2 = st.columns(2)
                ukc_se_dt_lo = _ss_number_input(_s1, "Odds 2x CI low (days)",
                    "ukc_superexp_dt_ci_lo", float(round(max(5.0, _seg_dt / 2))),
                    min_value=5.0, max_value=2000.0, step=5.0)
                ukc_se_dt_hi = _ss_number_input(_s2, "Odds 2x CI high (days)",
                    "ukc_superexp_dt_ci_hi", float(round(_seg_dt * 2)),
                    min_value=5.0, max_value=5000.0, step=5.0)
                if ukc_se_dt_lo > ukc_se_dt_hi:
                    st.error("Odds 2x CI low must be ≤ CI high.")
                    st.stop()
            else:
                _d1, _d2 = st.columns(2)
                ukc_dt_lo = _ss_number_input(_d1, "Odds 2x time CI low (days)",
                    "ukc_custom_dt_lo", float(round(max(5.0, _seg_dt / 2))),
                    min_value=5.0, max_value=2000.0, step=5.0,
                    help="Fast scenario: days for the odds p/(1-p) to double.")
                ukc_dt_hi = _ss_number_input(_d2, "Odds 2x time CI high (days)",
                    "ukc_custom_dt_hi", float(round(_seg_dt * 2)),
                    min_value=5.0, max_value=5000.0, step=5.0,
                    help="Slow scenario: days for the odds to double.")
                if ukc_dt_lo > ukc_dt_hi:
                    st.error("Odds 2x CI low must be ≤ CI high.")
                    st.stop()
                ukc_dt_dist = st.radio("Trend distribution",
                    ["Normal", "Lognormal", "Log-log"], horizontal=True,
                    key="ukc_custom_dt_dist")

            _p1, _p2 = st.columns(2)
            ukc_pos_lo = _ss_number_input(_p1, "Pos CI low (%)", "ukc_custom_pos_lo",
                round(max(_cur_score - 2.0, 0.1), 2), min_value=0.01, max_value=99.9, step=0.1)
            ukc_pos_hi = _ss_number_input(_p2, "Pos CI high (%)", "ukc_custom_pos_hi",
                round(min(_cur_score + 2.0, 99.9), 2), min_value=0.02, max_value=99.99, step=0.1)
            if ukc_pos_lo > ukc_pos_hi:
                st.error("Position CI low must be ≤ CI high.")
                st.stop()

        st.markdown("---")
        ukc_show_labels = st.toggle("Labels", key="ukc_labels")
        ukc_show_open = st.toggle("Open-weight models", key="ukc_show_open")
        ukc_show_lag = st.toggle("Lag markers", key="ukc_show_lag")

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox("Project as of", ukc_frontier_names,
                         index=ukc_as_of_idx, key='_ukc_proj_as_of',
                         help="Backtest: project from an earlier model's vantage point.")
            ukc_end_year = st.radio("Project through", [2026, 2027, 2028, 2029],
                                    horizontal=True, key="ukc_end_year")

    # ── Fit ──────────────────────────────────────────────────────────────
    frontier_used = ukc_frontier_all[:ukc_as_of_idx + 1]
    base_date = ukc_frontier_all[0]['date']
    days_used = np.array([(m['date'] - base_date).days for m in frontier_used], dtype=float)
    logit_used = _logit(np.array([m['cyber_score'] / 100 for m in frontier_used]))
    n_samples = N_SAMPLES

    _names_used = [m['name'] for m in frontier_used]
    _break_idxs = [_names_used.index(b) for b in ukc_bps if b in _names_used]
    _last_seg_start = _break_idxs[-1] if _break_idxs else 0

    current = frontier_used[-1]
    current_day = (current['date'] - base_date).days

    if ukc_is_superexp:
        z = 2 ** (days_used / ukc_halflife)
        X = np.column_stack([np.ones_like(z), z])
        (se_A, se_K), *_ = np.linalg.lstsq(X, logit_used, rcond=None)
        fitted_logit = se_A + se_K * 2 ** (current_day / ukc_halflife)
        proj_dt = _lognormal_from_ci(ukc_se_dt_lo, ukc_se_dt_hi, n_samples)
    else:
        _fit_days = days_used[_last_seg_start:]
        _fit_logit = logit_used[_last_seg_start:]
        if len(_fit_days) < 2:
            _fit_days, _fit_logit = days_used, logit_used
        seg_params = fit_line(_fit_days, _fit_logit)
        fitted_logit = seg_params[0] + seg_params[1] * current_day
        if ukc_dt_dist == "Log-log":
            proj_dt = _log_lognormal_from_ci(ukc_dt_lo, ukc_dt_hi, n_samples)
        elif ukc_dt_dist == "Lognormal":
            proj_dt = _lognormal_from_ci(ukc_dt_lo, ukc_dt_hi, n_samples)
        else:
            proj_dt = _normal_from_ci(ukc_dt_lo, ukc_dt_hi, n_samples)

    proj_logit_slope = np.log(2) / proj_dt

    # Position uncertainty, sampled in logit space so it respects the bound.
    _pos_lo_logit, _pos_hi_logit = _logit(ukc_pos_lo / 100), _logit(ukc_pos_hi / 100)
    _pos_sigma = max((_pos_hi_logit - _pos_lo_logit) / (2 * 1.282), 0)
    start_logit = np.random.normal(fitted_logit, _pos_sigma, n_samples)

    # ── Trajectories ─────────────────────────────────────────────────────
    proj_end_date = datetime(ukc_end_year, 12, 31)
    proj_n_days = max((proj_end_date - current['date']).days + 1, 2)
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates = [current['date'] + timedelta(days=int(d)) for d in proj_days_arr]

    if ukc_is_superexp:
        logit_traj = start_logit[:, None] + np.log(2) * superexp_trajectory(
            proj_days_arr, proj_dt, ukc_halflife, ukc_dt_floor)
    else:
        logit_traj = start_logit[:, None] + proj_days_arr[None, :] * proj_logit_slope[:, None]
    all_trajectories = _inv_logit(logit_traj) * 100

    pct = {p: np.percentile(all_trajectories, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}

    # ── Chart ────────────────────────────────────────────────────────────
    st.header("AISI Narrow Cyber Tasks — open-weight lag behind the closed frontier")

    # Computed before the chart because the callout below reports on it too.
    lag_rows = ukc_lag_rows(ukc_all, ukc_frontier_all)
    tlo_all, tlo_lag_rows = ukc_tlo_lag_rows()
    _render_ukcyber_newest_open(lag_rows, tlo_all, tlo_lag_rows)

    fig = go.Figure()

    for lo, hi, color, label in [
        (pct[5], pct[95], 'rgba(52,152,219,0.10)', '90% CI'),
        (pct[10], pct[90], 'rgba(52,152,219,0.18)', '80% CI'),
        (pct[25], pct[75], 'rgba(52,152,219,0.28)', '50% CI'),
    ]:
        fig.add_trace(go.Scatter(
            x=proj_dates + proj_dates[::-1],
            y=list(hi) + list(lo[::-1]),
            fill='toself', fillcolor=color, line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True,
        ))

    # Historical OLS through the fitted segment, then the projected slope.
    if not ukc_is_superexp:
        _d0, _d1x = int(_fit_days[0]), int(_fit_days[-1])
        _hd = np.arange(_d0, _d1x + 1, 1)
        _hy = _inv_logit(seg_params[0] + seg_params[1] * _hd) * 100
        _seg_dt_disp = np.log(2) / seg_params[1] if seg_params[1] > 0 else float('inf')
        _hdates = [base_date + timedelta(days=int(d)) for d in _hd]
        fig.add_trace(go.Scatter(
            x=_hdates, y=_hy.tolist(),
            mode='lines', line=dict(color='#2c3e50', width=2.5),
            name=f'Fitted trend (2x odds: {_seg_dt_disp:.0f}d)',
            hovertext=[f"{d.strftime('%b %d, %Y')}<br>Fitted trend: {y:.1f}%"
                       f"<br>Odds 2x: {_seg_dt_disp:.0f}d"
                       for d, y in zip(_hdates, _hy)],
            hoverinfo='text',
        ))

    fig.add_trace(go.Scatter(
        x=proj_dates, y=pct[50].tolist(),
        mode='lines', line=dict(color='#2980b9', width=2.5),
        name='Projection (median)',
        hovertext=[f"{d.strftime('%b %d, %Y')}<br>Median: {v:.1f}%"
                   for d, v in zip(proj_dates, pct[50])],
        hoverinfo='text',
    ))

    today = _add_today_vline(fig)

    # Target threshold the ETA below is measured against.
    fig.add_hline(y=_UKC_TARGET, line=dict(color='#e74c3c', width=1.5, dash='dot'),
                  opacity=0.8)
    fig.add_annotation(x=1.0, xref='paper', y=_UKC_TARGET, text=f'{_UKC_TARGET:.0f}%',
                       showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(size=10, color='#e74c3c'))

    # Backtesting against later frontier models.
    is_backtesting = ukc_as_of_idx < len(ukc_frontier_all) - 1
    backtest_results = []
    _bt_lookup = {}
    if is_backtesting:
        backtest_results = _backtest_stats(
            ukc_frontier_all[ukc_as_of_idx + 1:], all_trajectories,
            current['date'], proj_end_date,
            lambda m: m['cyber_score'], lambda m: m['name'])
        _bt_lookup = {r['name']: r for r in backtest_results}

    # Closed models that never set the frontier.
    for m in ukc_all:
        if m['is_frontier'] or m['weights'] != 'closed':
            continue
        fig.add_trace(go.Scatter(
            x=[m['date']], y=[m['cyber_score']],
            mode='markers' + ('+text' if ukc_show_labels else ''),
            marker=dict(color='#aaaaaa', size=7, symbol='circle-open',
                        line=dict(color='#bbbbbb', width=1)),
            text=[m['name']] if ukc_show_labels else None,
            textposition='top right', textfont=dict(size=8, color='#bbbbbb'),
            hovertext=f"{m['name']}<br>{m['date'].strftime('%b %d, %Y')}<br>"
                      f"Success: {m['cyber_score']:.1f}%<br>{m['organization']} · closed",
            hoverinfo='text', showlegend=False,
        ))

    # The closed frontier itself.
    for idx_m, m in enumerate(ukc_frontier_all):
        hover = (f"{m['name']}<br>{m['date'].strftime('%b %d, %Y')}<br>"
                 f"Success: {m['cyber_score']:.1f}%<br>{m['organization']} · closed")
        if idx_m <= ukc_as_of_idx:
            is_sel = idx_m == ukc_as_of_idx
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['cyber_score']],
                mode='markers' + ('+text' if ukc_show_labels else ''),
                marker=dict(color='#e74c3c' if is_sel else '#4F8DFD',
                            size=14 if is_sel else 10,
                            symbol='star' if is_sel else 'circle',
                            line=dict(color='white', width=1)),
                text=[m['name']] if ukc_show_labels else None,
                textposition='top right',
                textfont=dict(size=9, color='#c0392b' if is_sel else '#1a1a2e'),
                hovertext=hover, hoverinfo='text', showlegend=False,
            ))
        elif m['name'] in _bt_lookup:
            r = _bt_lookup[m['name']]
            _c = _bt_color_for(r)
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['cyber_score']],
                mode='markers+text',
                marker=dict(color=_c, size=12, symbol='diamond',
                            line=dict(color='white', width=1)),
                text=[f"{m['name']} (p{r['percentile']:.0f})"],
                textposition='top right', textfont=dict(size=9, color=_c),
                hovertext=hover + f"<br>Percentile: {r['percentile']:.0f}%",
                hoverinfo='text', showlegend=False,
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[m['date']], y=[m['cyber_score']],
                mode='markers' + ('+text' if ukc_show_labels else ''),
                marker=dict(color='#aaaaaa', size=10, symbol='circle-open',
                            line=dict(color='#777777', width=2)),
                text=[m['name']] if ukc_show_labels else None,
                textposition='top right', textfont=dict(size=9, color='#999999'),
                hovertext=hover, hoverinfo='text', showlegend=False,
            ))

    # Open-weight models, plus the horizontal lag connector back to the frontier.
    if ukc_show_open:
        _legend_seen = set()
        for r in lag_rows:
            _color = _UKC_OPEN_COLORS.get(r['country'], '#8e44ad')
            if r['lag_months'] is None:
                _lag_txt = "<br>Ahead of the closed frontier"
            else:
                _lag_txt = f"<br>Lag: {r['lag_months']:.1f} mo behind frontier"
                if r['lag_lo'] is not None and r['lag_hi'] is not None:
                    _lag_txt += (f"<br>Range: {r['lag_lo']:.1f}–{r['lag_hi']:.1f} mo"
                                 f"<br>(between {r['below_name']} and {r['above_name']})")
            if ukc_show_lag and r['match_date'] is not None:
                fig.add_trace(go.Scatter(
                    x=[r['match_date'], r['date']],
                    y=[r['cyber_score'], r['cyber_score']],
                    mode='lines', line=dict(color=_color, width=1.5, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ))
                fig.add_annotation(
                    x=r['match_date'] + (r['date'] - r['match_date']) / 2,
                    y=r['cyber_score'], text=f"{r['lag_months']:.1f} mo",
                    showarrow=False, yshift=-14,
                    font=dict(size=10, color=_color))
            _show_legend = r['country'] not in _legend_seen
            _legend_seen.add(r['country'])
            fig.add_trace(go.Scatter(
                x=[r['date']], y=[r['cyber_score']],
                mode='markers' + ('+text' if ukc_show_labels else ''),
                marker=dict(color=_color, size=12, symbol='square',
                            line=dict(color='white', width=1)),
                text=[r['name']] if ukc_show_labels else None,
                textposition='bottom right', textfont=dict(size=9, color=_color),
                name=f"Open-weight ({r['country']})",
                showlegend=_show_legend,
                hovertext=f"{r['name']}<br>{r['date'].strftime('%b %d, %Y')}<br>"
                          f"Success: {r['cyber_score']:.1f}%<br>"
                          f"{r['organization']} · {r['country']} · open{_lag_txt}",
                hoverinfo='text',
            ))

    if is_backtesting and backtest_results:
        _add_backtest_traces(fig, backtest_results, current['date'])

    _y_max = min(max(pct[95][-1], max(m['cyber_score'] for m in ukc_all) + 3, 60) + 5, 105)
    fig.update_layout(
        height=650,
        margin=dict(l=50, r=140, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(range=[ukc_all[0]['date'] - timedelta(days=30),
                          proj_end_date + timedelta(days=30)],
                   gridcolor='rgba(0,0,0,0.1)',
                   tickfont=dict(color='#1a1a2e'), zeroline=False),
        yaxis=dict(title="Avg success rate on 70 narrow cyber tasks (%)",
                   range=[0, _y_max], gridcolor='rgba(0,0,0,0.1)',
                   zeroline=False, ticksuffix='%',
                   tickfont=dict(color='#1a1a2e'), title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white',
    )

    st.plotly_chart(fig, width="stretch")
    if is_backtesting and backtest_results:
        _backtest_summary(backtest_results)

    # ── Time for open-weight models to reach the target ──────────────────
    st.subheader(f"Time for China to reach {_UKC_TARGET:.0f}%")

    eta = ukc_target_eta(ukc_all, ukc_frontier_all, _UKC_TARGET)
    if eta is None:
        st.info(f"The closed frontier has not yet reached {_UKC_TARGET:.0f}%, "
                "so there is no crossing date to lag behind.")
    else:
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.metric("Closed frontier crossed it",
                      eta['frontier_date'].strftime('%b %Y'))
            _btw = [n for n in eta['frontier_between'] if n]
            st.caption(" → ".join(_btw) if len(_btw) > 1 else (_btw[0] if _btw else ""))
        with _c2:
            st.metric("Measured open-weight lag",
                      f"{eta['lag_lo']:.1f}–{eta['lag_hi']:.1f} mo")
            st.caption("From the two open-weight models observed")
        with _c3:
            _lo, _hi = eta['date_lo'], eta['date_hi']
            _span = (f"{_lo.strftime('%b')}–{_hi.strftime('%b %Y')}"
                     if _lo.year == _hi.year else
                     f"{_lo.strftime('%b %Y')}–{_hi.strftime('%b %Y')}")
            st.metric(f"China reaches {_UKC_TARGET:.0f}%", _span)
            _m_lo = (_lo - today).days / _UKC_DAYS_PER_MONTH
            _m_hi = (_hi - today).days / _UKC_DAYS_PER_MONTH
            if _m_hi < 0:
                st.caption("Already passed")
            else:
                st.caption(f"{max(_m_lo, 0):.1f}–{max(_m_hi, 0):.1f} months from today")

        _direct = ukc_target_eta_direct(ukc_all, _UKC_TARGET)
        _direct_txt = (f" Extrapolating the two open-weight models directly instead gives "
                       f"{_direct.strftime('%b %Y')} — a fragile fit from two points "
                       f"53 days apart, shown only as a sanity check."
                       if _direct is not None else "")
        _btw_txt = (f" (interpolated between {eta['frontier_between'][0]} and "
                    f"{eta['frontier_between'][1]})"
                    if all(eta['frontier_between']) else "")
        _fn_caption(
            f"The closed frontier reached {_UKC_TARGET:.0f}% around "
            f"{eta['frontier_date'].strftime('%b %Y')}{_btw_txt}; open-weight models "
            f"have trailed it by {eta['lag_lo']:.1f}\u2013{eta['lag_hi']:.1f} "
            "months, so they reach it that much later." + _direct_txt,
            ("trailed it by", "Lags are measured against the frontier's interpolated "
                              "crossing of each score, not against the next model up "
                              "\u2014 snapping to the next model up would equate "
                              "scores several points apart."))

    # ── Cross-check: the same question on the long-horizon cyber range ───
    _render_ukcyber_tlo(lag_rows)

    _fn_caption(
        "Values are digitized from AISI's published figure \u2014 treat them as "
        "approximate. \u201cChina\u201d here is the two Chinese open-weight "
        "models. These are projections, not forecasts.",
        ("digitized from AISI's published figure", _UKC_PROVENANCE),
        ("\u201cChina\u201d here", _UKC_CONFOUND_PLAIN),
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))


def _render_ukcyber_newest_open(narrow_lag_rows, tlo_all, tlo_lag_rows):
    """Promote the newest open-weight model when only the cyber range has it.

    Without this the chart below silently omits the most recent open-weight
    release, and the only sign of it is a table at the bottom of the tab. Uses
    the lag bracket rather than the interpolated point estimate: TLO's frontier
    is sparse enough that the point estimate can sit anywhere inside a wide
    bracket, and the bracket is also what is comparable to the narrow-task
    figures elsewhere on the tab.
    """
    r = ukc_open_only_on_tlo(narrow_lag_rows, tlo_lag_rows)
    if r is None:
        return

    _color = _UKC_OPEN_COLORS.get(r['country'], '#8e44ad')
    _lag = (f"{r['lag_lo']:.1f}–{r['lag_hi']:.1f} mo" if r['lag_hi'] is not None
            else f"≥ {r['lag_lo']:.1f} mo")
    _fr = [m for m in tlo_all if m['is_frontier']]
    _best = max(_fr, key=lambda m: m['cyber_score']) if _fr else None

    with st.container(border=True):
        _c1, _c2, _c3 = st.columns(3)
        with _c1:
            st.markdown(f"**Newest open-weight model**<br>"
                        f"<span style='font-size:1.6rem;color:{_color}'>{r['name']}</span>",
                        unsafe_allow_html=True)
            st.caption(f"{r['organization']} · {r['country']} · "
                       f"released {r['date'].strftime('%b %d, %Y')}")
        with _c2:
            st.metric("Cyber range score", f"{r['cyber_score']:.1f} / {_UKC_TLO_STEPS} steps")
            st.caption(f"Closed frontier: {_best['cyber_score']:.1f} ({_best['name']})"
                       if _best else "")
        with _c3:
            st.metric("Lag behind closed frontier", _lag)
            _ends = [n for n in (r['above_name'], r['below_name']) if n]
            st.caption(f"Bracketed by {' and '.join(_ends)}" if _ends
                       else "Below every frontier model — lower bound only")

        _fn_caption(
            f"<b>Not on the chart below.</b> AISI/CAISI ran only a selective set on "
            f"{r['name']}, so it has no narrow-task score to plot \u2014 and the "
            "cyber range is the weaker of AISI's two measures.",
            ("a selective set", "ExploitBench and the \u201cThe Last Ones\u201d "
                                "cyber range \u2014 not the 70-task narrow suite "
                                "this chart is built from. It therefore does not "
                                "enter the projection or the lag readout beneath "
                                "it."),
            ("the weaker of AISI's two measures", "Its frontier is far sparser. See "
                                                  "the cross-check at the bottom of "
                                                  "this tab for the full "
                                                  "comparison."))


def _render_ukcyber_tlo(narrow_lag_rows):
    """AISI's other cyber measure, on the same models and the same lag method.

    Kept as a cross-check section rather than a second projection: TLO has only
    ten models and a much sparser frontier, so it is worth showing beside the
    narrow-task lag but not worth fanning out into its own forecast. Together
    the two reproduce AISI's "4 to 7 months" -- narrow tasks give the low end,
    the range gives the high end.
    """
    tlo_all, tlo_lag = ukc_tlo_lag_rows()
    tlo_frontier = [m for m in tlo_all if m['is_frontier']]
    if not tlo_lag:
        return

    st.subheader("Cross-check: long-horizon cyber range")
    _fn_caption(
        "AISI's other cyber measure: average steps completed on \u201cThe Last "
        f"Ones\u201d, a {_UKC_TLO_STEPS}-step simulated corporate-network "
        "attack. Same frontier and lag method as above, applied to steps.",
        ("average steps completed", "10 runs per model, 100M tokens each, scored "
                                    "on autonomous end-to-end execution rather "
                                    "than isolated skills."),
        ("Same frontier and lag method as above", "Applied to steps instead of "
                                                  "percent."))

    fig = go.Figure()
    _fx = [m['date'] for m in tlo_frontier]
    _fy = [m['cyber_score'] for m in tlo_frontier]
    fig.add_trace(go.Scatter(
        x=_fx, y=_fy, mode='lines+markers+text', name='Closed frontier',
        line=dict(color='#2c3e50', width=2), marker=dict(size=9, color='#2c3e50'),
        text=[m['name'] for m in tlo_frontier], textposition='top left',
        textfont=dict(size=9, color='#7f8c8d'),
        hovertext=[f"{m['name']}<br>{m['cyber_score']:.1f} of {_UKC_TLO_STEPS} steps"
                   f"<br>{m['date'].strftime('%b %d, %Y')}" for m in tlo_frontier],
        hoverinfo='text',
    ))
    _seen = set()
    for r in tlo_lag:
        _color = _UKC_OPEN_COLORS.get(r['country'], '#8e44ad')
        if r['match_date'] is not None:
            fig.add_trace(go.Scatter(
                x=[r['match_date'], r['date']], y=[r['cyber_score'], r['cyber_score']],
                mode='lines', line=dict(color=_color, width=1.5, dash='dot'),
                hoverinfo='skip', showlegend=False))
            fig.add_annotation(
                x=r['match_date'] + (r['date'] - r['match_date']) / 2,
                y=r['cyber_score'], text=f"{r['lag_months']:.1f} mo",
                showarrow=False, yshift=-14, font=dict(size=10, color=_color))
        fig.add_trace(go.Scatter(
            x=[r['date']], y=[r['cyber_score']], mode='markers+text',
            name=f"Open weight ({r['country']})",
            marker=dict(size=11, color=_color, symbol='diamond'),
            text=[r['name']], textposition='middle right',
            textfont=dict(size=9, color=_color),
            hovertext=[f"{r['name']}<br>{r['cyber_score']:.1f} of {_UKC_TLO_STEPS} steps"
                       f"<br>{r['lag_months']:.1f} mo behind the frontier"],
            hoverinfo='text', showlegend=r['country'] not in _seen))
        _seen.add(r['country'])

    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Release date", gridcolor='rgba(0,0,0,0.1)',
                   tickfont=dict(color='#1a1a2e'), zeroline=False),
        yaxis=dict(title=f"Avg steps completed (of {_UKC_TLO_STEPS})",
                   range=[0, _UKC_TLO_STEPS], gridcolor='rgba(0,0,0,0.1)',
                   zeroline=False, tickfont=dict(color='#1a1a2e'),
                   title_font=dict(color='#1a1a2e')),
        hovermode='closest',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)', font=dict(color='#1a1a2e')),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    st.plotly_chart(fig, width="stretch")

    # Side-by-side lag on the two measures, for models that appear on both.
    # Both columns use lag_lo (next model up) so they are on the same footing --
    # see the caption below for why the interpolated estimates are not.
    _narrow_by_name = {r['name']: r for r in narrow_lag_rows}
    rows = []
    for r in tlo_lag:
        _n = _narrow_by_name.get(r['name'])
        rows.append({
            'Model': r['name'],
            'Released': r['date'].strftime('%b %d, %Y'),
            'Steps': f"{r['cyber_score']:.1f} / {_UKC_TLO_STEPS}",
            'Lag — cyber range': (f"≥ {r['lag_lo']:.1f} mo" if r['lag_hi'] is None
                                  else f"{r['lag_lo']:.1f} mo"),
            'Lag — narrow tasks': (f"{_n['lag_lo']:.1f} mo"
                                   if _n and _n['lag_lo'] is not None else "not tested"),
        })
    st.table(rows)

    # Compared on lag_lo, the next-model-up convention AISI's own figure titles
    # use. The interpolated point estimates are not comparable across the two
    # datasets: DeepSeek-V4-Pro's narrow score falls in a 10-point frontier gap
    # while its TLO score sits below every frontier model, so on point estimates
    # the ordering flips for reasons about how each frontier is sampled.
    _both = [(r, _narrow_by_name.get(r['name'])) for r in tlo_lag]
    _both = [(a, b) for a, b in _both
             if b and b['lag_lo'] is not None and a['lag_lo'] is not None]
    if _both:
        _tlo_rng = (min(a['lag_lo'] for a, _ in _both),
                    max(a['lag_lo'] for a, _ in _both))
        _nar_rng = (min(b['lag_lo'] for _, b in _both),
                    max(b['lag_lo'] for _, b in _both))
        _fn_caption(
            f"On the models measured both ways the lag is "
            f"{_nar_rng[0]:.1f}\u2013{_nar_rng[1]:.1f} months on narrow tasks but "
            f"{_tlo_rng[0]:.1f}\u2013{_tlo_rng[1]:.1f} months on the cyber range "
            "\u2014 the two ends of AISI's \u201c4 to 7 months\u201d headline.",
            ("measured both ways", "Both figures use the next-model-up convention "
                                   "AISI's own chart titles use, which is the only "
                                   "one comparable across the two datasets."),
            ("the cyber range", "AISI treats it as the weaker evidence of the two: "
                                "it draws on far fewer tasks, and a model stalling "
                                "mid-chain may be failing on long-horizon planning "
                                "rather than on cyber skill. Only ten models have "
                                "been run on it, so the frontier it is measured "
                                "against is correspondingly coarse."))


# ── Revenue ──────────────────────────────────────────────────────────────

_OPENAI_REVENUE = [
    ("2022-12-31", 0.028),
    ("2023-03-01", 0.2),
    ("2023-12-31", 2.0),
    ("2024-08-01", 3.7),
    ("2024-12-31", 5.5),
    ("2025-03-01", 8.0),
    ("2025-06-01", 10.0),
    ("2025-07-01", 12.0),
    ("2025-08-01", 13.0),
    ("2025-12-31", 21.4),
    ("2026-02-01", 25.0),
    ("2026-03-31", 26.0),
    ("2026-04-30", 28.0),
    ("2026-05-15", 30.0),
    ("2026-05-27", 33.0),
    # TickerTrends' 2026-07-30 post ("OpenAI ARR Growth Accelerated Into July")
    # prints its underlying read dates rather than a month label, so these four
    # carry real as-of dates instead of a publication date. Same alt-data series
    # as everything from 2025-12-31 down.
    ("2026-06-25", 37.3),
    ("2026-07-02", 38.5),
    ("2026-07-09", 40.3),
    # TickerTrends alt-data estimate (published 2026-07-23), dated to publication
    # since that post gives only "July 2026". Not a disclosure and uncorroborated by
    # press -- mainstream reporting still cites ~$25B as of Feb 2026. Kept because
    # the 21.4 and 33.0 points above are from the same TickerTrends series, so this
    # continues one line rather than splicing two.
    ("2026-07-23", 41.3),
    ("2026-07-29", 42.6),
    # TickerTrends' 2026-08-14 post ("OpenAI ARR Tracking Reached $44.3B as Bloomberg
    # Reported $40B Run Rate") states "$44.3B as of Aug. 12, up from $42.6B on Jul. 29",
    # so it chains explicitly off the row above and carries a real as-of date.
    ("2026-08-12", 44.3),
    # Excluded: Bloomberg reported 2026-08-13 that OpenAI's run rate "topped $40B" as of
    # 2026-07-31 (sourced to internal staff memos). A press report, not the TickerTrends
    # alt-data series -- and it sits *below* the 42.6 tracked two days earlier, the same
    # gross-vs-net gap that makes splicing the two source classes produce fake dips.
    # Epoch's ai_companies_revenue_reports.csv carries that $40B as its newest OpenAI row;
    # we deliberately keep the alt-data line instead of switching mid-series.
    # Rechecked 2026-08-21: TickerTrends has published nothing with a company-wide OpenAI
    # total since the 2026-08-14 post that produced the 44.3 above, so this series is
    # current. Greg Brockman's internal note (July run rate grew >20% MoM) states no
    # absolute figure and is not a datapoint. Also excluded from the 2026-08-18 TickerTrends
    # post: Codex $8.83B tracked ARR (week ending 08-10) is product-level, not a company
    # total. CFO Sarah Friar's 2026-08-14 closed-door investor meeting (CNBC) cited only
    # Bloomberg's older, lower $40B; its news was the enterprise/consumer mix crossover.
    # Rechecked 2026-08-29: TickerTrends has published no company-wide OpenAI total
    # since the 2026-08-14 post behind the 44.3 above; its three posts since (08-21
    # enterprise adoption, 08-26 Codex/Claude Code adoption, 08-26 GitHub commit
    # mentions) carry adoption counts and no dollar figures. Epoch's
    # ai_companies_revenue_reports.csv still tops out at Bloomberg's $40B / 2026-08-13.
    # Excluded: OpenAI CFO Sarah Friar told employees on 2026-07-29 that July ARR
    # "surpassed the company's entire second quarter" (CNBC). No dollar figure was
    # given, and it mixes ARR against a quarterly total -- not a datapoint.
    # Rechecked 2026-09-03: still nothing company-wide since the 2026-08-14 post behind the
    # 44.3 above -- TickerTrends' six posts since (08-21 enterprise adoption, 08-26 Codex/Claude
    # Code adoption, 08-26 GitHub commit mentions, 08-31 GTA VI, 08-31 ChatGPT-ads ARR, ~09-02
    # Fable 5.1 launch tracking) carry no company total. Epoch's ai_companies_revenue_reports.csv
    # still tops out at Bloomberg's $40B / 2026-07-31 as-of.
    # NOTE for the next run: TickerTrends began a *weekly* ChatGPT Ads ARR series on 2026-08-31
    # ("$1 billion in latest reported ARR"). That is product-level, like Codex ARR, and must not
    # be read as a company total -- it will generate frequent "OpenAI ARR" headlines that do not
    # belong in this table.
]

_ANTHROPIC_REVENUE = [
    ("2022-12-31", 0.01),
    ("2023-12-31", 0.1),
    ("2024-12-31", 1.0),
    ("2025-01-01", 1.0),
    ("2025-05-01", 3.0),
    ("2025-06-01", 4.0),
    ("2025-08-01", 5.0),
    ("2025-10-01", 7.0),
    ("2025-12-31", 9.0),
    ("2026-02-01", 14.0),
    ("2026-03-01", 19.0),
    ("2026-04-01", 30.0),
    # Anthropic's Series H post (announced 05-28) said the run rate "crossed
    # $47 billion earlier this month", so the figure is dated to mid-May rather
    # than to the announcement. The earlier 40.0 (05-15) and 45.0 (05-21) media
    # reports were dropped: both are superseded by that company disclosure and
    # would otherwise sit at or below it on the same or a later date.
    ("2026-05-15", 47.0),
    # SOURCE-CLASS EXCEPTION, added 2026-08-18 with explicit user sign-off: this is a
    # *press report of a private investor update*, not a public company disclosure --
    # the only non-disclosed point in this series. Bloomberg (2026-08-17) reported the
    # run rate "surpasses $65 billion", sourced to people familiar with a regular
    # investor update; CNBC ran it as "Anthropic tells investors annualized revenue run
    # rate climbed to $65 billion in July". Dated 2026-07-31 (end of July), which is the
    # as-of date, matching how Epoch's ai_companies_revenue_reports.csv carries the row --
    # not the 08-17 publication date. anthropic.com/news has no August revenue post; the
    # newest *public* company statement is still the Series H $47B above.
    # Admitted because the figure is company-originated and is the right unit (annualized
    # run rate), so it is a much smaller basis shift than a tracker estimate would be.
    # Note it is still a steep bend -- 47 -> 65 in 2.5 months -- so treat any projection
    # anchored on it with care. If Anthropic states a run rate publicly (the S-1 announced
    # 2026-06-01 is still a confidential draft, nothing filed on EDGAR), prefer that.
    ("2026-07-31", 65.0),
    # Every OTHER point in this series is company-disclosed, deliberately. Third-party tracker
    # estimates exist for mid-2026 (YipitData $69B at 2026-07-10; TickerTrends $69.6B
    # June / $74.1B July) but are not added: TickerTrends put April at $35.6B where
    # Anthropic disclosed $30B, so appending one would bend the curve by a source-level
    # shift rather than by measured growth. Anthropic's "exceed $50B by end of July"
    # investor guidance is a forecast, not an achieved run rate, and is excluded too;
    # so is the $10.9B Q2-2026 figure shown to investors -- a projection, and quarterly
    # revenue rather than run rate. Rechecked 2026-08-18: still no *public*
    # company disclosure since the Series H $47B -- the 65.0 above came via press. The
    # TickerTrends line now reads $69.6B (June) / $74.1B (July), i.e. ~14% above the
    # investor-reported 65.0, which is the same tracker-vs-disclosure gap seen in April
    # and the reason that series is still excluded.
    # Rechecked 2026-08-29: still nothing newer. EDGAR full-text search returns no
    # Anthropic S-1 and no operating-company registrant (the 35 "anthropic" names are
    # investor SPVs and feeder funds; the 2026-06-01 S-1 is still confidential), so the
    # anticipated late-August public filing has not happened. anthropic.com/news has no
    # revenue post, and Epoch's newest Anthropic row is the same 65.0 / 2026-07-31.
    # Also excluded, and prominent in mid-August press: Q2-2026 revenue "over $11.5B"
    # (Bloomberg 2026-08-14, CNBC/Fortune 2026-08-15). Quarterly revenue, not run rate,
    # and sourced to people familiar rather than disclosed. Naively annualizing lands
    # near the May 47.0 by coincidence; that is not a reason to enter it.
    # Also excluded (2026-08-18 check): full-year 2026 guidance of $100-120B and 2028
    # guidance of $190-200B are forecasts; ARK's 2026-08-18 "Anthropic + OpenAI revenue
    # tops $115B" is a third-party combined aggregate, not a per-company run rate.
    # Rechecked 2026-08-21: still nothing newer. anthropic.com/news has no revenue post
    # (newest items are 08-14 text watermark, 08-07 Fable 5 safeguards, 08-04 Cuellar hire),
    # and EDGAR still shows no Anthropic PBC registrant -- the 2026-06-01 S-1 remains a
    # confidential submission, so no public prospectus figure exists. Epoch's
    # ai_companies_revenue_reports.csv newest Anthropic row is the same 65.0 / 2026-07-31.
    # All 08-17-onward coverage (CNBC, Axios, TechCrunch, Fortune) is derivative of the one
    # Bloomberg scoop, not independent reads. TickerTrends' 2026-08-18 post gives Claude Code
    # $15.12B tracked ARR and calls it "21.9% of Anthropic's total tracked ARR"; that implies
    # ~$69B but TickerTrends never states the total, and it would be tracker-class anyway.
    # Rechecked 2026-09-03: still nothing newer. anthropic.com/news has nine posts since 08-04
    # (through the 09-01 Fable 5.1 / Mythos 5.1 launch) and none states a run rate -- the launch
    # post is product/pricing only, unlike the April Google/Broadcom post that carried the $30B.
    # EDGAR still has no Anthropic registrant at all (a company search for S-1 returns no
    # matching companies; the 137 "anthropic" hits since 08-01 are third-party NPORT-P fund
    # holdings), so the 2026-06-01 S-1 remains a confidential draft and no public prospectus
    # figure exists. Epoch's newest Anthropic row is unchanged at 65.0 / 2026-07-31, sourced to
    # the same Bloomberg story. All later coverage (Fortune, CNBC, TechCrunch, Axios) is
    # derivative of that one scoop, not an independent read.
]


def _parse_revenue(data):
    dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in data]
    values = [v for _, v in data]
    return dates, values


def _rev_value_at(dates, vals, d):
    """A revenue series' value on date `d`: log-linear between the bracketing
    reports, held flat past the last one (never extrapolated), None before the
    first."""
    if d < dates[0]:
        return None
    if d >= dates[-1]:
        return vals[-1]
    i = next(i for i in range(1, len(dates)) if d <= dates[i])
    if d == dates[i - 1]:
        return vals[i - 1]
    f = (d - dates[i - 1]).days / (dates[i] - dates[i - 1]).days
    return float(2.0 ** (np.log2(vals[i - 1]) * (1 - f) + np.log2(vals[i]) * f))


def _rev_combined_series(a_dates, a_vals, b_dates, b_vals):
    """Sum two ARR series onto the union of their report dates.

    The two companies are reported on different days, so each side is
    interpolated to the other's dates by `_rev_value_at`. Starts where both
    have data, so the sum never steps up merely because a series began.
    """
    start = max(a_dates[0], b_dates[0])
    out_dates = [d for d in sorted(set(a_dates) | set(b_dates)) if d >= start]
    out_vals = [_rev_value_at(a_dates, a_vals, d) + _rev_value_at(b_dates, b_vals, d)
                for d in out_dates]
    return out_dates, out_vals


def _rev_fit_and_project(dates, vals, n_recent, proj_end, n_samples=None,
                          dt_lo_override=None, dt_hi_override=None):
    """Fit exponential (OLS in log-space) to last n_recent points,
    sample doubling-time fan chart, return (proj_dates, percentiles_dict, ols_dt, ols_dates, ols_vals)."""
    if n_samples is None:
        n_samples = N_SAMPLES
    log_vals = np.log2(np.array(vals))
    base = dates[0]
    days = np.array([(d - base).days for d in dates], dtype=float)

    # Fit on last n_recent points
    fit_days = days[-n_recent:]
    fit_log = log_vals[-n_recent:]
    params = fit_line(fit_days, fit_log)
    # slope = log2(revenue) per day → doubling time = 1/slope days
    ols_dt = 1.0 / params[1] if params[1] > 0 else 365

    # OLS trend line through all data (fit is on last N, but draw through full range)
    d0, d1 = int(days[0]), int(days[-1])
    ols_d = np.arange(d0, d1 + 1)
    ols_log = params[0] + params[1] * ols_d
    ols_dates_out = [base + timedelta(days=int(d)) for d in ols_d]
    ols_vals_out = 2.0 ** ols_log

    # Projection from last data point
    last_date = dates[-1]
    last_log = params[0] + params[1] * days[-1]  # fitted value at last point
    proj_n_days = (proj_end - last_date).days + 1
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates_out = [last_date + timedelta(days=int(d)) for d in proj_days_arr]

    # Doubling time CI
    dt_lo = dt_lo_override if dt_lo_override is not None else max(10, ols_dt * 0.65)
    dt_hi = dt_hi_override if dt_hi_override is not None else ols_dt * 1.5
    sampled_dt = _lognormal_from_ci(dt_lo, dt_hi, n_samples)

    # Position noise in log-space (±0.3 log2 units ≈ ±23% revenue)
    pos_sigma = 0.3
    start_log = np.random.normal(last_log, pos_sigma, n_samples)

    # Build trajectories in log2-space, convert to linear
    trajectories = 2.0 ** (start_log[:, None] + proj_days_arr[None, :] / sampled_dt[:, None])

    pcts = {}
    for p in [5, 10, 25, 50, 75, 90, 95]:
        pcts[p] = np.percentile(trajectories, p, axis=0)

    return proj_dates_out, pcts, ols_dt, dt_lo, dt_hi, ols_dates_out, ols_vals_out, trajectories


def _fmt_revenue(val):
    """Format revenue value in $B to human-readable string."""
    if val >= 1000:
        return f"${val/1000:.1f}T"
    if val >= 1:
        return f"${val:.1f}B"
    return f"${val*1000:.0f}M"


_REV_COMBINED_NAME = "OpenAI + Anthropic"
_REV_COMBINED_COLOR = '#5b6ab0'

_REV_MILESTONES = [
    (50, "$50B"),
    (100, "$100B"),
    (200, "$200B"),
    (500, "$500B"),
    (1000, "$1T"),
]


def render_revenue():
    st.header("Revenue Projections (ARR)")
    st.warning("Warning: projections assume lines go up indefinitely, which is "
               "obviously not realistic.")

    openai_dates, openai_vals = _parse_revenue(_OPENAI_REVENUE)
    anthropic_dates, anthropic_vals = _parse_revenue(_ANTHROPIC_REVENUE)

    # Build "project as of" options from combined dates
    _all_rev_dates = sorted(set(openai_dates + anthropic_dates))
    _rev_date_labels = []
    for _d in _all_rev_dates:
        _lbl = _d.strftime('%b %Y')
        _parts = []
        _oai_v = next((v for dd, v in zip(openai_dates, openai_vals) if dd == _d), None)
        _ant_v = next((v for dd, v in zip(anthropic_dates, anthropic_vals) if dd == _d), None)
        if _oai_v is not None:
            _parts.append(f"OAI {_fmt_revenue(_oai_v)}")
        if _ant_v is not None:
            _parts.append(f"Ant {_fmt_revenue(_ant_v)}")
        _rev_date_labels.append(f"{_lbl} ({', '.join(_parts)})")

    _rev_as_of_label = st.session_state.get('rev_proj_as_of', _rev_date_labels[-1])
    if _rev_as_of_label not in _rev_date_labels:
        _rev_as_of_label = _rev_date_labels[-1]
    _rev_as_of_idx = _rev_date_labels.index(_rev_as_of_label)
    _rev_as_of_date = _all_rev_dates[_rev_as_of_idx]

    # Filter data to "as of" cutoff for fitting
    oai_dates_fit = [d for d in openai_dates if d <= _rev_as_of_date]
    oai_vals_fit = [v for d, v in zip(openai_dates, openai_vals) if d <= _rev_as_of_date]
    ant_dates_fit = [d for d in anthropic_dates if d <= _rev_as_of_date]
    ant_vals_fit = [v for d, v in zip(anthropic_dates, anthropic_vals) if d <= _rev_as_of_date]
    comb_dates, comb_vals = _rev_combined_series(
        openai_dates, openai_vals, anthropic_dates, anthropic_vals)
    comb_dates_fit = [d for d in comb_dates if d <= _rev_as_of_date]
    comb_vals_fit = [v for d, v in zip(comb_dates, comb_vals) if d <= _rev_as_of_date]
    _rev_backtesting = _rev_as_of_idx < len(_all_rev_dates) - 1
    oai_can_project = len(oai_vals_fit) >= 3
    ant_can_project = len(ant_vals_fit) >= 3
    comb_can_project = len(comb_vals_fit) >= 3

    # Clamp slider session state values if they exceed filtered data length
    if st.session_state.get('oai_n_recent', 0) > len(oai_vals_fit):
        st.session_state.pop('oai_n_recent', None)
    if st.session_state.get('ant_n_recent', 0) > len(ant_vals_fit):
        st.session_state.pop('ant_n_recent', None)
    if st.session_state.get('comb_n_recent', 0) > len(comb_vals_fit):
        st.session_state.pop('comb_n_recent', None)

    with st.sidebar:
        st.header("Revenue Projection")
        rev_end_year = st.radio(
            "Project through", [2026, 2027, 2028, 2029],
            horizontal=True, key="rev_end_year", index=0)
        log_scale = st.checkbox("Log scale", value=True, key="rev_log_scale")
        rev_2025_only = st.checkbox("2025+ only", value=False, key="rev_2025_only")
        show_milestones = st.toggle("Milestones", value=True, key="rev_milestones")
        show_labels = st.toggle("Labels", value=True, key="rev_labels")
        show_combined = st.toggle(
            "Combined line", value=False, key="rev_combined",
            help="Add OpenAI + Anthropic as one series, fitted and projected "
                 "like the other two. The two companies report on different "
                 "days, so each is interpolated to the other's dates (held "
                 "flat past its own last report) before summing.")

        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                _rev_date_labels,
                index=_rev_date_labels.index(_rev_as_of_label),
                key='rev_proj_as_of',
                help="Backtest: project from an earlier date's vantage point.",
            )

        with st.expander("OpenAI projection"):
            if oai_can_project:
                # Default to fitting all available points (maximum); user can reduce.
                oai_n_recent = st.slider("Fit to last N points", 3, len(oai_vals_fit),
                                          value=len(oai_vals_fit), key="oai_n_recent")
                # Pre-compute OLS DT for defaults
                _oai_log = np.log2(np.array(oai_vals_fit))
                _oai_days = np.array([(d - oai_dates_fit[0]).days for d in oai_dates_fit], dtype=float)
                _oai_p = fit_line(_oai_days[-oai_n_recent:], _oai_log[-oai_n_recent:])
                _oai_ols_dt = 1.0 / _oai_p[1] if _oai_p[1] > 0 else 200
                _oai_dt_lo_def = float(round(max(10, _oai_ols_dt * 0.65)))
                _oai_dt_hi_def = float(round(_oai_ols_dt * 1.5))
                oai_dt_lo = _ss_number_input(st, "DT CI low (days)", "oai_dt_lo",
                                              _oai_dt_lo_def, min_value=5.0, step=5.0)
                oai_dt_hi = _ss_number_input(st, "DT CI high (days)", "oai_dt_hi",
                                              _oai_dt_hi_def, min_value=10.0, step=5.0)
            else:
                st.caption("Need ≥3 data points to project")

        with st.expander("Anthropic projection"):
            if ant_can_project:
                # Default to fitting all available points (maximum); user can reduce.
                ant_n_recent = st.slider("Fit to last N points", 3, len(ant_vals_fit),
                                          value=len(ant_vals_fit), key="ant_n_recent")
                _ant_log = np.log2(np.array(ant_vals_fit))
                _ant_days = np.array([(d - ant_dates_fit[0]).days for d in ant_dates_fit], dtype=float)
                _ant_p = fit_line(_ant_days[-ant_n_recent:], _ant_log[-ant_n_recent:])
                _ant_ols_dt = 1.0 / _ant_p[1] if _ant_p[1] > 0 else 100
                _ant_dt_lo_def = float(round(max(10, _ant_ols_dt * 0.65)))
                _ant_dt_hi_def = float(round(_ant_ols_dt * 1.5))
                ant_dt_lo = _ss_number_input(st, "DT CI low (days)", "ant_dt_lo",
                                              _ant_dt_lo_def, min_value=5.0, step=5.0)
                ant_dt_hi = _ss_number_input(st, "DT CI high (days)", "ant_dt_hi",
                                              _ant_dt_hi_def, min_value=10.0, step=5.0)
            else:
                st.caption("Need ≥3 data points to project")

        comb_n_recent = len(comb_vals_fit)
        comb_dt_lo = comb_dt_hi = None
        if show_combined:
            with st.expander("Combined projection"):
                if comb_can_project:
                    comb_n_recent = st.slider("Fit to last N points", 3, len(comb_vals_fit),
                                              value=len(comb_vals_fit), key="comb_n_recent")
                    _comb_log = np.log2(np.array(comb_vals_fit))
                    _comb_days = np.array([(d - comb_dates_fit[0]).days for d in comb_dates_fit],
                                          dtype=float)
                    _comb_p = fit_line(_comb_days[-comb_n_recent:], _comb_log[-comb_n_recent:])
                    _comb_ols_dt = 1.0 / _comb_p[1] if _comb_p[1] > 0 else 200
                    comb_dt_lo = _ss_number_input(st, "DT CI low (days)", "comb_dt_lo",
                                                  float(round(max(10, _comb_ols_dt * 0.65))),
                                                  min_value=5.0, step=5.0)
                    comb_dt_hi = _ss_number_input(st, "DT CI high (days)", "comb_dt_hi",
                                                  float(round(_comb_ols_dt * 1.5)),
                                                  min_value=10.0, step=5.0)
                else:
                    st.caption("Need ≥3 data points to project")

    proj_end = datetime(rev_end_year, 12, 31)

    # Filter display data for 2025+ if toggled (projections still use all data for fitting)
    _rev_cutoff = datetime(2025, 1, 1) if rev_2025_only else datetime(2000, 1, 1)
    _oai_display = [(d, v) for d, v in zip(openai_dates, openai_vals) if d >= _rev_cutoff]
    _ant_display = [(d, v) for d, v in zip(anthropic_dates, anthropic_vals) if d >= _rev_cutoff]
    oai_display_dates = [d for d, _ in _oai_display]
    oai_display_vals = [v for _, v in _oai_display]
    ant_display_dates = [d for d, _ in _ant_display]
    ant_display_vals = [v for _, v in _ant_display]
    _comb_display = [(d, v) for d, v in zip(comb_dates, comb_vals) if d >= _rev_cutoff]
    comb_display_dates = [d for d, _ in _comb_display]
    comb_display_vals = [v for _, v in _comb_display]
    x_start = min(oai_display_dates[0], ant_display_dates[0]) - timedelta(days=30)

    oai_proj_dates = oai_pcts = oai_dt = oai_ols_dates = oai_ols_vals = oai_traj = None
    ant_proj_dates = ant_pcts = ant_dt = ant_ols_dates = ant_ols_vals = ant_traj = None
    comb_proj_dates = comb_pcts = comb_dt = comb_ols_dates = comb_ols_vals = comb_traj = None

    if oai_can_project:
        oai_proj_dates, oai_pcts, oai_dt, oai_dt_lo_eff, oai_dt_hi_eff, oai_ols_dates, oai_ols_vals, oai_traj = \
            _rev_fit_and_project(oai_dates_fit, oai_vals_fit, oai_n_recent, proj_end,
                                  dt_lo_override=oai_dt_lo, dt_hi_override=oai_dt_hi)

    if ant_can_project:
        ant_proj_dates, ant_pcts, ant_dt, ant_dt_lo_eff, ant_dt_hi_eff, ant_ols_dates, ant_ols_vals, ant_traj = \
            _rev_fit_and_project(ant_dates_fit, ant_vals_fit, ant_n_recent, proj_end,
                                  dt_lo_override=ant_dt_lo, dt_hi_override=ant_dt_hi)

    if show_combined and comb_can_project:
        comb_proj_dates, comb_pcts, comb_dt, comb_dt_lo_eff, comb_dt_hi_eff, \
            comb_ols_dates, comb_ols_vals, comb_traj = \
            _rev_fit_and_project(comb_dates_fit, comb_vals_fit, comb_n_recent, proj_end,
                                  dt_lo_override=comb_dt_lo, dt_hi_override=comb_dt_hi)

    # Backtest stats
    oai_bt_results = []
    ant_bt_results = []
    comb_bt_results = []
    if _rev_backtesting:
        if oai_can_project:
            oai_future = [{'date': d, 'value': v}
                          for d, v in zip(openai_dates, openai_vals) if d > _rev_as_of_date]
            oai_bt_results = _backtest_stats(
                oai_future, oai_traj, oai_dates_fit[-1], proj_end,
                lambda m: m['value'], lambda m: f"OAI {m['date'].strftime('%b %Y')}",
            )
        if ant_can_project:
            ant_future = [{'date': d, 'value': v}
                          for d, v in zip(anthropic_dates, anthropic_vals) if d > _rev_as_of_date]
            ant_bt_results = _backtest_stats(
                ant_future, ant_traj, ant_dates_fit[-1], proj_end,
                lambda m: m['value'], lambda m: f"Ant {m['date'].strftime('%b %Y')}",
            )
        if show_combined and comb_traj is not None:
            comb_future = [{'date': d, 'value': v}
                           for d, v in zip(comb_dates, comb_vals) if d > _rev_as_of_date]
            comb_bt_results = _backtest_stats(
                comb_future, comb_traj, comb_dates_fit[-1], proj_end,
                lambda m: m['value'], lambda m: f"Sum {m['date'].strftime('%b %Y')}",
            )

    fig = go.Figure()

    # --- Fan bands ---
    _bt_results_map = {"OpenAI": oai_bt_results, "Anthropic": ant_bt_results,
                       _REV_COMBINED_NAME: comb_bt_results}
    _can_project_map = {"OpenAI": oai_can_project, "Anthropic": ant_can_project,
                        _REV_COMBINED_NAME: comb_can_project}

    _rev_series = [
        ("OpenAI", oai_proj_dates, oai_pcts, '#10a37f', oai_display_dates, oai_display_vals,
         oai_ols_dates, oai_ols_vals, oai_dt),
        ("Anthropic", ant_proj_dates, ant_pcts, '#d4a574', ant_display_dates, ant_display_vals,
         ant_ols_dates, ant_ols_vals, ant_dt),
    ]
    if show_combined:
        _rev_series.append(
            (_REV_COMBINED_NAME, comb_proj_dates, comb_pcts, _REV_COMBINED_COLOR,
             comb_display_dates, comb_display_vals, comb_ols_dates, comb_ols_vals, comb_dt))

    _rev_companies = []
    for _name, _proj_dates, _pcts, _color, _disp_dates, _disp_vals, \
            _ols_dates_raw, _ols_vals_raw, _dt in _rev_series:
        if not _can_project_map[_name]:
            # No projection — just show data points
            _rev_companies.append((_name, None, None, _color, _disp_dates, _disp_vals,
                                   None, None, None))
            continue
        _ols_disp = [(d, v) for d, v in zip(_ols_dates_raw, _ols_vals_raw) if d >= _rev_cutoff]
        _rev_companies.append((_name, _proj_dates, _pcts, _color, _disp_dates, _disp_vals,
                               [d for d, _ in _ols_disp], [v for _, v in _ols_disp], _dt))

    for name, p_dates, pcts, base_color, data_dates, data_vals, \
            ols_dates, ols_vals, dt in _rev_companies:
        # Parse base color to rgba
        r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)

        if pcts is not None:
            bands = [
                (pcts[5], pcts[95], f'rgba({r},{g},{b},0.08)', f'{name} 90% CI'),
                (pcts[10], pcts[90], f'rgba({r},{g},{b},0.15)', f'{name} 80% CI'),
                (pcts[25], pcts[75], f'rgba({r},{g},{b},0.25)', f'{name} 50% CI'),
            ]
            for lo, hi, color, label in bands:
                x_poly = p_dates + p_dates[::-1]
                y_poly = list(hi) + list(lo[::-1])
                fig.add_trace(go.Scatter(
                    x=x_poly, y=y_poly,
                    fill='toself', fillcolor=color,
                    line=dict(width=0),
                    name=label, hoverinfo='skip', showlegend=False,
                ))

            # Median projection line
            fig.add_trace(go.Scatter(
                x=p_dates, y=list(pcts[50]),
                mode='lines',
                line=dict(color=base_color, width=2, dash='dash'),
                name=f'{name} median projection',
                hovertemplate='%{{x|%b %Y}}<br>{name}: %{{y:.1f}}B<extra></extra>'.format(name=name),
            ))

            # OLS trend line
            fig.add_trace(go.Scatter(
                x=ols_dates, y=list(ols_vals),
                mode='lines',
                line=dict(color=base_color, width=2),
                name=f'{name} trend (DT={dt:.0f}d)',
                hoverinfo='skip', showlegend=True,
            ))

        # Data points — split into used vs future when backtesting
        bt_results = _bt_results_map[name]
        if _rev_backtesting:
            used_dates = [d for d in data_dates if d <= _rev_as_of_date]
            used_vals = [v for d, v in zip(data_dates, data_vals) if d <= _rev_as_of_date]
            future_dates = [d for d in data_dates if d > _rev_as_of_date]
            future_vals = [v for d, v in zip(data_dates, data_vals) if d > _rev_as_of_date]

            # Used points — normal style
            if used_dates:
                hover_texts = [f"{name}<br>{d.strftime('%b %Y')}<br>{_fmt_revenue(v)}"
                               for d, v in zip(used_dates, used_vals)]
                fig.add_trace(go.Scatter(
                    x=used_dates, y=used_vals,
                    mode='markers' + ('+text' if show_labels else ''),
                    marker=dict(color=base_color, size=9, line=dict(color='white', width=1)),
                    text=[_fmt_revenue(v) for v in used_vals] if show_labels else None,
                    textposition='top right',
                    textfont=dict(size=8, color=base_color),
                    hovertext=hover_texts, hoverinfo='text',
                    name=name, showlegend=True,
                ))

            # Future points — backtest colored diamonds
            bt_lookup = {r['date']: r for r in bt_results}
            for d, v in zip(future_dates, future_vals):
                bt_r = bt_lookup.get(d)
                pt_color = _bt_color_for(bt_r) if bt_r else '#aaaaaa'
                pct_label = f"P{bt_r['percentile']:.0f}" if bt_r else ""
                hover = f"{name}<br>{d.strftime('%b %Y')}<br>{_fmt_revenue(v)}"
                if bt_r:
                    hover += f"<br>{pct_label}"
                fig.add_trace(go.Scatter(
                    x=[d], y=[v],
                    mode='markers+text',
                    marker=dict(color=pt_color, size=11, symbol='diamond',
                                line=dict(color='white', width=1)),
                    text=pct_label,
                    textposition='top right',
                    textfont=dict(size=8, color=pt_color),
                    hovertext=hover, hoverinfo='text',
                    showlegend=False,
                ))
        else:
            hover_texts = [f"{name}<br>{d.strftime('%b %Y')}<br>{_fmt_revenue(v)}"
                           for d, v in zip(data_dates, data_vals)]
            fig.add_trace(go.Scatter(
                x=data_dates, y=data_vals,
                mode='markers' + ('+text' if show_labels else ''),
                marker=dict(color=base_color, size=9, line=dict(color='white', width=1)),
                text=[_fmt_revenue(v) for v in data_vals] if show_labels else None,
                textposition='top right',
                textfont=dict(size=8, color=base_color),
                hovertext=hover_texts, hoverinfo='text',
                name=name, showlegend=True,
            ))

    # --- Milestone hlines ---
    if show_milestones:
        x_lo = x_start
        x_hi = proj_end
        _ms_colors = ['#888888', '#666666', '#c0392b', '#8e44ad', '#2c3e50']
        for (val, label), color in zip(_REV_MILESTONES, _ms_colors):
            fig.add_trace(go.Scatter(
                x=[x_lo, x_hi], y=[val, val],
                mode='lines', line=dict(color=color, width=1.2, dash='dot'),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_annotation(
                x=1.0, xref='paper', y=val, text=f"  {label}",
                showarrow=False, xanchor='left', yanchor='middle',
                font=dict(size=10, color=color))

    # --- Backtest traces ---
    if _rev_backtesting:
        fig.add_vline(x=_rev_as_of_date, line=dict(color='#e67e22', width=2, dash='dash'), opacity=0.8)
        fig.add_annotation(
            x=_rev_as_of_date, y=1.0, yref='paper',
            text='  Projection start', showarrow=False, textangle=-90,
            font=dict(size=10, color='#e67e22'), xanchor='right', yanchor='top',
        )
        for bt_results, bt_color in [(oai_bt_results, '#10a37f'), (ant_bt_results, '#d4a574'),
                                     (comb_bt_results, _REV_COMBINED_COLOR)]:
            if len(bt_results) >= 2:
                fig.add_trace(go.Scatter(
                    x=[r['date'] for r in bt_results],
                    y=[r['value'] for r in bt_results],
                    mode='lines',
                    line=dict(color=bt_color, width=2, dash='dash'),
                    hoverinfo='skip', showlegend=False,
                ))

    today = _add_today_vline(fig)

    # --- Layout ---
    yaxis_type = "log" if log_scale else "linear"
    all_display_vals = oai_display_vals + ant_display_vals
    if show_combined:
        all_display_vals = all_display_vals + comb_display_vals
    y_min_data = min(all_display_vals)
    # Use 90th pctile (not 95th) to set range — 95th can be absurdly large
    _y_max_parts = []
    if oai_pcts is not None:
        _y_max_parts.append(oai_pcts[90][-1])
    if ant_pcts is not None:
        _y_max_parts.append(ant_pcts[90][-1])
    if comb_pcts is not None:
        _y_max_parts.append(comb_pcts[90][-1])
    y_max_proj = max(_y_max_parts) if _y_max_parts else max(all_display_vals)
    y_max = max(max(all_display_vals), y_max_proj) * 1.5

    fig.update_layout(
        yaxis_title="ARR ($ Billions)",
        yaxis_type=yaxis_type,
        xaxis_title="",
        xaxis_range=[x_start, proj_end + timedelta(days=30)],
        height=600,
        template="plotly_white",
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
    )
    if log_scale:
        tickvals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]
        ticktext = ["$10M", "$30M", "$100M", "$300M", "$1B", "$3B", "$10B", "$30B",
                     "$100B", "$300B", "$1T", "$3T"]
        fig.update_yaxes(
            tickvals=tickvals, ticktext=ticktext,
            range=[np.log10(y_min_data * 0.5), np.log10(y_max)],
        )
    else:
        fig.update_yaxes(range=[0, y_max])

    st.plotly_chart(fig, width="stretch")

    # --- Backtest summary ---
    if _rev_backtesting:
        all_bt = oai_bt_results + ant_bt_results
        if all_bt:
            _backtest_summary(all_bt)

    # --- Milestone arrival estimates ---
    # A milestone already reached is dropped. The floor is per series, so
    # switching the combined line on never hides a company's own rows.
    _rev_reached = max(oai_display_vals + ant_display_vals)
    if show_milestones:
        with st.expander("Milestone details"):
            _ms_rows = [("OpenAI", oai_proj_dates, oai_traj, _rev_reached),
                        ("Anthropic", ant_proj_dates, ant_traj, _rev_reached)]
            if show_combined:
                _ms_rows.append((_REV_COMBINED_NAME, comb_proj_dates, comb_traj,
                                 max(comb_display_vals)))
            for name, p_dates, traj, reached in _ms_rows:
                if p_dates is None or traj is None:
                    continue
                arrival_rows = []
                for val, label in _REV_MILESTONES:
                    if val <= reached:
                        continue
                    # For each trajectory, find first day it crosses val
                    crossed = np.argmax(traj >= val, axis=1)
                    # argmax returns 0 if never crossed — check if actually crossed
                    actually_crossed = traj[np.arange(len(traj)), crossed] >= val
                    if actually_crossed.sum() < len(traj) * 0.05:
                        arrival_rows.append({"Milestone": label, "Median": "Beyond range", "50% CI": "—", "80% CI": "—"})
                        continue
                    # Among those that crossed, get dates
                    crossed_days = crossed[actually_crossed]
                    p50_day = int(np.percentile(crossed_days, 50))
                    p25_day = int(np.percentile(crossed_days, 25))
                    p75_day = int(np.percentile(crossed_days, 75))
                    p10_day = int(np.percentile(crossed_days, 10))
                    p90_day = int(np.percentile(crossed_days, 90))
                    base = p_dates[0]
                    p50_date = base + timedelta(days=p50_day)
                    p25_date = base + timedelta(days=p25_day)
                    p75_date = base + timedelta(days=p75_day)
                    p10_date = base + timedelta(days=p10_day)
                    p90_date = base + timedelta(days=p90_day)
                    arrival_rows.append({
                        "Milestone": label,
                        "Median": p50_date.strftime('%b %Y'),
                        "50% CI": f"{p25_date.strftime('%b %Y')} – {p75_date.strftime('%b %Y')}",
                        "80% CI": f"{p10_date.strftime('%b %Y')} – {p90_date.strftime('%b %Y')}",
                    })
                if arrival_rows:
                    st.markdown(f"**{name}**")
                    st.table(arrival_rows)

    # --- Doubling times ---
    with st.expander("Historical doubling times"):
        col1, col2 = st.columns(2)
        for col, name, dates, vals in [
            (col1, "OpenAI", openai_dates, openai_vals),
            (col2, "Anthropic", anthropic_dates, anthropic_vals),
        ]:
            with col:
                st.markdown(f"**{name}**")
                rows = []
                for i in range(1, len(vals)):
                    if vals[i] > 0 and vals[i - 1] > 0 and vals[i] > vals[i - 1]:
                        days = (dates[i] - dates[i - 1]).days
                        growth = vals[i] / vals[i - 1]
                        if growth > 1 and days > 0:
                            doubling_days = days * np.log(2) / np.log(growth)
                            rows.append({
                                "Period": f"{dates[i-1].strftime('%b %Y')} → {dates[i].strftime('%b %Y')}",
                                "Growth": f"{growth:.1f}x",
                                "Doubling Time": f"{doubling_days:.0f} days",
                            })
                if rows:
                    st.table(rows)

    _fn_caption(
        "Figures are approximate ARR compiled from public reports and media "
        "sources. These are projections, not forecasts.",
        ("public reports and media sources",
         "Anthropic's Dec 2025 figure is averaged from a reported $8\u201310B "
         "range. May 2026 figures from The Information "
         "(theinformation.com/articles/"
         "anthropic-openais-share-ai-startup-revenues-rises-89 and "
         "/openai-held-1-billion-revenue-lead-anthropic-first-quarter)."),
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))


# ── Employment ────────────────────────────────────────────────────────────

_EMP_RESET_KEYS = [
    "emp_custom_dt_lo", "emp_custom_dt_hi",
    "emp_custom_pos_lo", "emp_custom_pos_hi",
    "emp_piecewise_n_seg", "emp_bp1_select", "emp_bp2_select",
    "emp_custom_dt_dist", "emp_custom_pos_dist",
    "emp_superexp_dt_init", "emp_superexp_halflife",
    "emp_superexp_dt_floor", "emp_superexp_dt_ci_lo",
    "emp_superexp_dt_ci_hi", "emp_superexp_pos_lo",
    "emp_superexp_pos_hi",
    "emp_proj_basis",
    "_emp_proj_as_of", "emp_end_year",
    "_emp_seg_config",
    "emp_rli_coverage", "emp_supervision_overhead",
    "emp_remote_digital_share", "emp_base_unemployment",
    "emp_jevons_recovery",
    "emp_base_unemp_lo", "emp_base_unemp_hi",
    "emp_jevons_lo", "emp_jevons_hi",
    "emp_adoption_lag", "emp_lag_lo", "emp_lag_hi",
    "emp_display_mode", "emp_labor_force",
    "emp_breakdown_date",
]

_EMP_DEFAULTS = {
    "emp_proj_basis": "Linear (logit)",
    "emp_piecewise_n_seg": 1,
    "emp_custom_dt_dist": "Lognormal",
    "emp_custom_pos_dist": "Normal",
    "emp_end_year": 2028,
    "emp_rli_coverage": 70.0,
    "emp_supervision_overhead": 10.0,
    "emp_remote_digital_share": 38.0,
    "emp_base_unemployment": 4.0,
    "emp_jevons_recovery": 30.0,
    "emp_base_unemp_lo": 2.5,
    "emp_base_unemp_hi": 5.5,
    "emp_jevons_lo": 15.0,
    "emp_jevons_hi": 45.0,
    "emp_adoption_lag": 365.0,
    "emp_lag_lo": 180.0,
    "emp_lag_hi": 730.0,
    "emp_display_mode": "Unemployment Rate (%)",
    "emp_labor_force": 167.0,
}


def render_employment():
    if st.session_state.pop("_reset_emp", False):
        for k in _EMP_RESET_KEYS:
            st.session_state.pop(k, None)
        st.session_state.update(_EMP_DEFAULTS)
        st.rerun()

    for k, v in _EMP_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Employment Model")

        emp_proj_as_of_name = st.session_state.get('_emp_proj_as_of', rli_frontier_names[-1])
        if emp_proj_as_of_name not in rli_frontier_names:
            emp_proj_as_of_name = rli_frontier_names[-1]
        emp_proj_as_of_idx = rli_frontier_names.index(emp_proj_as_of_name)

        # --- RLI Projection basis ---
        st.subheader("RLI Projection")
        emp_basis_options = ["Linear (logit)", "Piecewise linear (logit)", "Superexponential (logit)"]
        emp_proj_basis = st.radio("Projection basis", emp_basis_options, key="emp_proj_basis",
                                  help="Controls how the RLI score is projected forward.")

        emp_custom_dt_lo = emp_custom_dt_hi = None
        emp_custom_pos_lo = emp_custom_pos_hi = None
        emp_custom_dt_dist = "Lognormal"
        emp_custom_pos_dist = "Normal"
        emp_piecewise_n_segments = 1
        emp_piecewise_breakpoints = []
        _emp_is_linear = emp_proj_basis in ("Linear (logit)", "Piecewise linear (logit)")
        if emp_proj_basis == "Piecewise linear (logit)":
            emp_piecewise_n_segments = 2

        # Pre-compute OLS DT for defaults
        _emp_pre_fr = rli_frontier_all[:emp_proj_as_of_idx + 1]
        _emp_pre_base = rli_frontier_all[0]['date']
        _emp_pre_days = np.array([(m['date'] - _emp_pre_base).days for m in _emp_pre_fr], dtype=float)
        _emp_pre_logit = _logit(np.array([m['rli_score'] / 100 for m in _emp_pre_fr]))
        _emp_pre_params = fit_line(_emp_pre_days, _emp_pre_logit) if len(_emp_pre_fr) >= 2 else np.array([0, 0.007])
        _emp_pre_dt = round(np.log(2) / _emp_pre_params[1]) if _emp_pre_params[1] > 0 else 100

        if _emp_is_linear:
            with st.expander("RLI advanced options"):
                st.button("Reset to defaults", key="reset_emp_linear",
                          on_click=lambda: st.session_state.update(_reset_emp=True))

                _emp_bp_names = [m['name'] for m in rli_frontier_all[:emp_proj_as_of_idx + 1]]
                if emp_proj_basis == "Piecewise linear (logit)":
                    _emp_seg_options = [1, 2, 3] if len(_emp_bp_names) >= 5 else [1, 2]
                    if st.session_state.get("emp_piecewise_n_seg", 1) < 2:
                        st.session_state["emp_piecewise_n_seg"] = 2
                    emp_piecewise_n_segments = st.radio(
                        "Segments", _emp_seg_options,
                        horizontal=True, key="emp_piecewise_n_seg")
                else:
                    emp_piecewise_n_segments = 1
                    st.session_state.pop("emp_piecewise_n_seg", None)
                if emp_piecewise_n_segments >= 2:
                    _emp_default_bp1 = _emp_bp_names[len(_emp_bp_names) // 2]
                    _emp_bp1_idx = _emp_bp_names.index(_emp_default_bp1) if _emp_default_bp1 in _emp_bp_names else len(_emp_bp_names) // 2
                    emp_bp1_name = st.selectbox(
                        "Breakpoint", _emp_bp_names[1:],
                        index=max(0, _emp_bp1_idx - 1), key="emp_bp1_select")
                    emp_piecewise_breakpoints.append(emp_bp1_name)
                if emp_piecewise_n_segments >= 3:
                    _emp_bp1_pos = _emp_bp_names.index(emp_bp1_name)
                    _emp_remaining = _emp_bp_names[_emp_bp1_pos + 1:]
                    emp_bp2_name = st.selectbox(
                        "Breakpoint 2", _emp_remaining[:-1],
                        index=len(_emp_remaining[:-1]) // 2, key="emp_bp2_select")
                    emp_piecewise_breakpoints.append(emp_bp2_name)

                # DT defaults from last segment
                if emp_piecewise_n_segments >= 2 and emp_piecewise_breakpoints:
                    _emp_last_bp_idx = _emp_bp_names.index(emp_piecewise_breakpoints[-1]) if emp_piecewise_breakpoints[-1] in _emp_bp_names else 0
                    _emp_pw_seg_days = _emp_pre_days[_emp_last_bp_idx:]
                    _emp_pw_seg_logit = _emp_pre_logit[_emp_last_bp_idx:]
                    if len(_emp_pw_seg_days) >= 2:
                        _emp_pw_seg_params = fit_line(_emp_pw_seg_days, _emp_pw_seg_logit)
                        _emp_pw_seg_dt = round(np.log(2) / _emp_pw_seg_params[1]) if _emp_pw_seg_params[1] > 0 else _emp_pre_dt
                    else:
                        _emp_pw_seg_dt = _emp_pre_dt
                    _emp_default_dt_lo = float(round(max(5.0, _emp_pw_seg_dt / 2), 0))
                    _emp_default_dt_hi = float(round(_emp_pw_seg_dt * 2, 0))
                else:
                    _emp_default_dt_lo = float(round(max(5.0, _emp_pre_dt / 2), 0))
                    _emp_default_dt_hi = float(round(_emp_pre_dt * 2, 0))

                # Auto-update DT CIs when segment config changes
                _emp_seg_config = (emp_piecewise_n_segments, tuple(emp_piecewise_breakpoints))
                if st.session_state.get("_emp_seg_config") != _emp_seg_config:
                    st.session_state["_emp_seg_config"] = _emp_seg_config
                    st.session_state.pop("emp_custom_dt_lo", None)
                    st.session_state.pop("emp_custom_dt_hi", None)

                _emp_dt_lo_col, _emp_dt_hi_col = st.columns(2)
                emp_custom_dt_lo = _ss_number_input(_emp_dt_lo_col,
                    "Odds 2x time CI low (days)", "emp_custom_dt_lo", _emp_default_dt_lo,
                    min_value=5.0, max_value=2000.0, step=5.0)
                emp_custom_dt_hi = _ss_number_input(_emp_dt_hi_col,
                    "Odds 2x time CI high (days)", "emp_custom_dt_hi", _emp_default_dt_hi,
                    min_value=5.0, max_value=5000.0, step=5.0)
                if emp_custom_dt_lo > emp_custom_dt_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()

                _emp_cur = rli_frontier_all[emp_proj_as_of_idx]
                _emp_def_score = _emp_cur['rli_score']
                _emp_pos_lo_col, _emp_pos_hi_col = st.columns(2)
                emp_custom_pos_lo = _ss_number_input(_emp_pos_lo_col,
                    "Pos CI low (%)", "emp_custom_pos_lo", round(max(_emp_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1)
                emp_custom_pos_hi = _ss_number_input(_emp_pos_hi_col,
                    "Pos CI high (%)", "emp_custom_pos_hi", round(_emp_def_score + 1.0, 2),
                    step=0.1)

                emp_custom_dt_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"],
                    horizontal=True, key="emp_custom_dt_dist")
                emp_custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal"],
                    horizontal=True, key="emp_custom_pos_dist")

        # --- Superexponential controls ---
        emp_superexp_dt_initial = emp_superexp_halflife = None
        emp_superexp_dt_ci_lo = emp_superexp_dt_ci_hi = None
        emp_superexp_pos_lo = emp_superexp_pos_hi = None
        emp_superexp_dt_floor = 10
        emp_is_superexp = False
        if emp_proj_basis == "Superexponential (logit)":
            emp_is_superexp = True
            _emp_default_dt_init = 100.0
            if len(rli_frontier_all[:emp_proj_as_of_idx + 1]) >= 2:
                _emp_base = rli_frontier_all[0]['date']
                _emp_fr = rli_frontier_all[:emp_proj_as_of_idx + 1]
                _emp_fd = np.array([(m['date'] - _emp_base).days for m in _emp_fr], dtype=float)
                _emp_flogit = _logit(np.array([m['rli_score'] / 100 for m in _emp_fr]))
                _emp_fp = fit_line(_emp_fd, _emp_flogit)
                if _emp_fp[1] > 0:
                    _emp_default_dt_init = round(np.log(2) / _emp_fp[1], 0)

            _emp_pre_se_halflife = 365
            _emp_pre_se_z = 2 ** (_emp_pre_days / _emp_pre_se_halflife)
            _emp_pre_se_X = np.column_stack([np.ones_like(_emp_pre_se_z), _emp_pre_se_z])
            (_emp_pre_se_A, _emp_pre_se_K), *_ = np.linalg.lstsq(_emp_pre_se_X, _emp_pre_logit, rcond=None)
            _emp_pre_se_d_last = _emp_pre_days[-1]
            if _emp_pre_se_K > 0:
                _emp_pre_se_logit_slope = _emp_pre_se_K * np.log(2) * 2 ** (_emp_pre_se_d_last / _emp_pre_se_halflife) / _emp_pre_se_halflife
                _emp_pre_se_dt = round(np.log(2) / _emp_pre_se_logit_slope, 0)
            else:
                _emp_pre_se_dt = _emp_pre_dt
            _emp_default_se_dt_lo = float(round(max(5.0, _emp_pre_se_dt / 2), 0))
            _emp_default_se_dt_hi = float(round(_emp_pre_se_dt * 2, 0))

            with st.expander("RLI advanced options"):
                st.button("Reset to defaults", key="reset_emp_superexp",
                          on_click=lambda: st.session_state.update(_reset_emp=True))
                _emp_se_col1, _emp_se_col2 = st.columns(2)
                emp_superexp_dt_initial = _ss_number_input(_emp_se_col1,
                    "Initial odds 2x time (days)", "emp_superexp_dt_init", _emp_default_dt_init,
                    min_value=5.0, max_value=2000.0, step=5.0)
                emp_superexp_halflife = _ss_number_input(_emp_se_col2,
                    "Rate half-life (days)", "emp_superexp_halflife", 365,
                    min_value=30, max_value=5000, step=30)
                emp_superexp_dt_floor = _ss_number_input(st,
                    "Min odds 2x time (days)", "emp_superexp_dt_floor", 15.0,
                    min_value=1.0, max_value=500.0, step=1.0)
                _emp_se_ci1, _emp_se_ci2 = st.columns(2)
                emp_superexp_dt_ci_lo = _ss_number_input(_emp_se_ci1,
                    "Odds 2x CI low (days)", "emp_superexp_dt_ci_lo", _emp_default_se_dt_lo,
                    min_value=5.0, max_value=2000.0, step=5.0)
                emp_superexp_dt_ci_hi = _ss_number_input(_emp_se_ci2,
                    "Odds 2x CI high (days)", "emp_superexp_dt_ci_hi", _emp_default_se_dt_hi,
                    min_value=5.0, max_value=5000.0, step=5.0)
                if emp_superexp_dt_ci_lo > emp_superexp_dt_ci_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()
                _emp_cur = rli_frontier_all[emp_proj_as_of_idx]
                _emp_def_score = _emp_cur['rli_score']
                _emp_se_pos1, _emp_se_pos2 = st.columns(2)
                emp_superexp_pos_lo = _ss_number_input(_emp_se_pos1,
                    "Pos CI low (%)", "emp_superexp_pos_lo", round(max(_emp_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1)
                emp_superexp_pos_hi = _ss_number_input(_emp_se_pos2,
                    "Pos CI high (%)", "emp_superexp_pos_hi", round(_emp_def_score + 1.0, 2),
                    step=0.1)

        # --- Economic model parameters ---
        st.markdown("---")
        st.subheader("Economic Model")
        st.button("Reset to defaults", key="reset_emp_all",
                  on_click=lambda: st.session_state.update(_reset_emp=True))
        emp_rli_coverage = st.slider("RLI Coverage of Remote/Digital Work (%)",
                                      0.0, 100.0, key="emp_rli_coverage",
                                      help="Fraction of remote/digital task-hours the RLI benchmark represents.")
        emp_supervision = st.slider("AI Supervision Overhead (%)",
                                     0.0, 50.0, key="emp_supervision_overhead",
                                     help="New overhead for human QA/supervision of AI outputs.")
        emp_remote_share = st.slider("Remote/Digital Share of US Jobs (%)",
                                      0.0, 100.0, key="emp_remote_digital_share",
                                      help="Fraction of all US jobs that are remote/digital.")
        emp_base_unemp = st.slider("Base Unemployment Rate (%)",
                                    0.0, 15.0, key="emp_base_unemployment",
                                    help="Baseline unemployment rate before AI displacement.")
        emp_jevons = st.slider("Jevons/Reallocation Recovery (%)",
                                0.0, 100.0, key="emp_jevons_recovery",
                                help="How much displacement gets absorbed by Jevons paradox & reallocation.")
        emp_lag_days = st.slider("Adoption Lag (days)",
                                  0.0, 1460.0, key="emp_adoption_lag", step=30.0,
                                  help="Delay between AI capability and labor market impact. Default ~1 year.")

        st.markdown("---")
        st.subheader("Display")
        emp_display_mode = st.radio("Show as", ["Unemployment Rate (%)", "Jobs Lost Above Baseline"],
                                     horizontal=True, key="emp_display_mode")
        if emp_display_mode == "Jobs Lost Above Baseline":
            emp_labor_force = _ss_number_input(st,
                "US Labor Force (millions)", "emp_labor_force", 167.0,
                min_value=50.0, max_value=500.0, step=1.0,
                help="US civilian labor force size in millions.")
        else:
            emp_labor_force = st.session_state.get("emp_labor_force", 167.0)

        with st.expander("Uncertainty parameters"):
            _emp_bu_c1, _emp_bu_c2 = st.columns(2)
            emp_base_unemp_lo = _ss_number_input(_emp_bu_c1,
                "Base unemp CI low (%)", "emp_base_unemp_lo", 2.5,
                min_value=0.0, max_value=15.0, step=0.5,
                help="80% CI at 1 year. Uncertainty starts at 0 and grows with √time.")
            emp_base_unemp_hi = _ss_number_input(_emp_bu_c2,
                "Base unemp CI high (%)", "emp_base_unemp_hi", 5.5,
                min_value=0.0, max_value=15.0, step=0.5,
                help="80% CI at 1 year. Uncertainty starts at 0 and grows with √time.")
            _emp_jv_c1, _emp_jv_c2 = st.columns(2)
            emp_jevons_lo = _ss_number_input(_emp_jv_c1,
                "Jevons CI low (%)", "emp_jevons_lo", 15.0,
                min_value=0.0, max_value=100.0, step=5.0)
            emp_jevons_hi = _ss_number_input(_emp_jv_c2,
                "Jevons CI high (%)", "emp_jevons_hi", 45.0,
                min_value=0.0, max_value=100.0, step=5.0)
            _emp_lag_c1, _emp_lag_c2 = st.columns(2)
            emp_lag_lo = _ss_number_input(_emp_lag_c1,
                "Lag CI low (days)", "emp_lag_lo", 180.0,
                min_value=0.0, max_value=1460.0, step=30.0)
            emp_lag_hi = _ss_number_input(_emp_lag_c2,
                "Lag CI high (days)", "emp_lag_hi", 730.0,
                min_value=0.0, max_value=1460.0, step=30.0)

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                rli_frontier_names,
                index=rli_frontier_names.index(emp_proj_as_of_name),
                key='_emp_proj_as_of',
                help="Project from an earlier model's vantage point.",
            )
            _emp_end_year = st.radio(
                "Project through", [2026, 2027, 2028, 2029],
                horizontal=True, key="emp_end_year")

    # ── Build RLI data arrays ────────────────────────────────────────────
    emp_frontier_used = rli_frontier_all[:emp_proj_as_of_idx + 1]
    base_date = rli_frontier_all[0]['date']
    days_all = np.array([(m['date'] - base_date).days for m in rli_frontier_all], dtype=float)
    logit_all = _logit(np.array([m['rli_score'] / 100 for m in rli_frontier_all]))

    _emp_fit_end = emp_proj_as_of_idx + 1
    days_used = days_all[:_emp_fit_end]
    logit_used = logit_all[:_emp_fit_end]
    n_used = len(emp_frontier_used)

    n_emp = N_SAMPLES
    if emp_proj_basis in ("Linear (logit)", "Piecewise linear (logit)"):
        if emp_piecewise_n_segments >= 2:
            _emp_bp_names_used = [m['name'] for m in emp_frontier_used]
            _emp_seg_break_idxs = []
            for bp_name in emp_piecewise_breakpoints:
                if bp_name in _emp_bp_names_used:
                    _emp_seg_break_idxs.append(_emp_bp_names_used.index(bp_name))
            _emp_last_seg_start = _emp_seg_break_idxs[-1] if _emp_seg_break_idxs else 0
            _emp_last_seg_range = list(range(_emp_last_seg_start, n_used))
            _emp_params = fit_line(days_used[_emp_last_seg_range], logit_used[_emp_last_seg_range])
        else:
            _emp_params = fit_line(days_used, logit_used)

        _emp_current_day = (emp_frontier_used[-1]['date'] - base_date).days
        if emp_piecewise_n_segments >= 2:
            _emp_seg_d = days_used[_emp_last_seg_range]
            _emp_seg_y = logit_used[_emp_last_seg_range]
        else:
            _emp_seg_d = days_used
            _emp_seg_y = logit_used
        _emp_intercept = np.mean(_emp_seg_y - _emp_params[1] * _emp_seg_d)
        _emp_fitted_logit = _emp_intercept + _emp_params[1] * _emp_current_day

        if emp_custom_dt_dist == "Log-log":
            emp_proj_dt = _log_lognormal_from_ci(emp_custom_dt_lo, emp_custom_dt_hi, n_emp)
        elif emp_custom_dt_dist == "Lognormal":
            emp_proj_dt = _lognormal_from_ci(emp_custom_dt_lo, emp_custom_dt_hi, n_emp)
        else:
            emp_proj_dt = _normal_from_ci(emp_custom_dt_lo, emp_custom_dt_hi, n_emp)

        emp_proj_logit_slope = np.log(2) / emp_proj_dt

        if emp_custom_pos_dist == "Lognormal":
            _emp_pos_logit_lo = _logit(emp_custom_pos_lo / 100)
            _emp_pos_logit_hi = _logit(emp_custom_pos_hi / 100)
            _emp_pos_offset = 10
            _emp_pos_sigma = (np.log(_emp_pos_logit_hi + _emp_pos_offset) - np.log(_emp_pos_logit_lo + _emp_pos_offset)) / (2 * 1.282)
            _emp_pos_mu = np.log(_emp_fitted_logit + _emp_pos_offset)
            emp_proj_start_logit = np.random.lognormal(_emp_pos_mu, max(_emp_pos_sigma, 0), n_emp) - _emp_pos_offset
        else:
            _emp_pos_logit_lo = _logit(emp_custom_pos_lo / 100)
            _emp_pos_logit_hi = _logit(emp_custom_pos_hi / 100)
            _emp_pos_sigma = (_emp_pos_logit_hi - _emp_pos_logit_lo) / (2 * 1.282)
            emp_proj_start_logit = np.random.normal(_emp_fitted_logit, max(_emp_pos_sigma, 0), n_emp)

    elif emp_proj_basis == "Superexponential (logit)":
        _emp_se_days = np.array([(m['date'] - base_date).days for m in emp_frontier_used], dtype=float)
        _emp_se_logit = _logit(np.array([m['rli_score'] / 100 for m in emp_frontier_used]))
        _emp_se_z = 2 ** (_emp_se_days / emp_superexp_halflife)
        _emp_se_X = np.column_stack([np.ones_like(_emp_se_z), _emp_se_z])
        (_emp_se_A, _emp_se_K), *_ = np.linalg.lstsq(_emp_se_X, _emp_se_logit, rcond=None)
        _emp_se_current_day = (emp_frontier_used[-1]['date'] - base_date).days
        _emp_se_fitted_logit = _emp_se_A + _emp_se_K * 2 ** (_emp_se_current_day / emp_superexp_halflife)

        emp_proj_dt = _lognormal_from_ci(emp_superexp_dt_ci_lo, emp_superexp_dt_ci_hi, n_emp)
        emp_proj_logit_slope = np.log(2) / emp_proj_dt

        _emp_se_pos_logit_lo = _logit(emp_superexp_pos_lo / 100)
        _emp_se_pos_logit_hi = _logit(emp_superexp_pos_hi / 100)
        _emp_se_pos_sigma = (_emp_se_pos_logit_hi - _emp_se_pos_logit_lo) / (2 * 1.282)
        emp_proj_start_logit = np.random.normal(_emp_se_fitted_logit, max(_emp_se_pos_sigma, 0), n_emp)

    # ── Build RLI trajectories ───────────────────────────────────────────
    emp_current = emp_frontier_used[-1]
    proj_end_date = datetime(_emp_end_year, 12, 31)
    proj_n_days = (proj_end_date - emp_current['date']).days + 1
    proj_days_arr = np.arange(0, proj_n_days, 1)
    proj_dates = [emp_current['date'] + timedelta(days=int(d)) for d in proj_days_arr]

    n_samples = len(emp_proj_dt)
    if emp_is_superexp:
        all_logit_traj = emp_proj_start_logit[:, None] + np.log(2) * superexp_trajectory(
            proj_days_arr, emp_proj_dt, emp_superexp_halflife, emp_superexp_dt_floor)
    else:
        all_logit_traj = emp_proj_start_logit[:, None] + proj_days_arr[None, :] * emp_proj_logit_slope[:, None]

    # RLI scores as fraction (0-1) for each sample at each timestep
    rli_traj_frac = _inv_logit(all_logit_traj)
    del all_logit_traj  # free (n_samples, n_days) array

    # ── Apply adoption lag ───────────────────────────────────────────────
    # Sample lag per trajectory: at time t, effective RLI = RLI(t - lag).
    # For t < lag, look up historical RLI score from frontier data.
    _lag_sigma = (emp_lag_hi - emp_lag_lo) / (2 * 1.282)
    lag_samples = np.clip(
        np.random.normal(emp_lag_days, max(_lag_sigma, 0), n_samples), 0, None).astype(int)
    max_lag = int(np.max(lag_samples)) if len(lag_samples) > 0 else 0

    # Build daily historical RLI (fraction) for up to max_lag days before projection start.
    # At proj day t with lag L, effective date = emp_current['date'] + t - L.
    # For t < L, that date is before the projection start, so we need history.
    # _hist_rli_daily[d] = RLI fraction at (emp_current['date'] - max_lag + d).
    _hist_rli_daily = np.full(max_lag, 0.004)  # floor: 0.4% before first indexed model
    if max_lag > 0:
        for _hm in rli_frontier_all:
            _hm_offset = (emp_current['date'] - _hm['date']).days  # days before projection start
            if _hm_offset < 0:
                break  # model is after projection start
            # This model covers days from (max_lag - _hm_offset) onward in the history array
            _hist_start = max_lag - _hm_offset
            if _hist_start < max_lag:
                _hist_rli_daily[max(0, _hist_start):] = _hm['rli_score'] / 100

    n_timesteps = rli_traj_frac.shape[1]
    rli_traj_lagged = np.empty_like(rli_traj_frac)
    for i in range(n_samples):
        lag_i = lag_samples[i]
        if lag_i <= 0:
            rli_traj_lagged[i] = rli_traj_frac[i]
        else:
            # For t < lag_i: effective date is before projection start, use historical RLI
            _fill_n = min(lag_i, n_timesteps)
            _start = max_lag - lag_i
            rli_traj_lagged[i, :_fill_n] = _hist_rli_daily[_start:_start + _fill_n]
            # For t >= lag_i: effective date is within projection, use shifted projection
            if lag_i < n_timesteps:
                rli_traj_lagged[i, lag_i:] = rli_traj_frac[i, :n_timesteps - lag_i]

    # ── Apply economic displacement model ────────────────────────────────
    # Base unemployment: known today, uncertainty grows with sqrt(time).
    # CI bounds define 80% CI at 1 year out; sigma scales as sqrt(days/365).
    _bu_sigma_1yr = (emp_base_unemp_hi - emp_base_unemp_lo) / (2 * 1.282)
    _bu_z = np.random.normal(0, 1, n_samples)                 # (n_samples,)
    _bu_time_scale = np.sqrt(proj_days_arr / 365.0)            # (n_timesteps,)
    base_unemp_arr = np.clip(
        emp_base_unemp + _bu_z[:, None] * _bu_sigma_1yr * _bu_time_scale[None, :],
        0, 15) / 100                                           # (n_samples, n_timesteps)

    _jv_sigma = (emp_jevons_hi - emp_jevons_lo) / (2 * 1.282)
    jevons_samples = np.clip(
        np.random.normal(emp_jevons, _jv_sigma, n_samples), 0, 100) / 100

    rli_cov = emp_rli_coverage / 100
    supervision = emp_supervision / 100
    remote_share = emp_remote_share / 100

    # Collapse displacement cascade into minimal intermediate arrays:
    #   disrupted = lagged_rli * coverage
    #   worker_occupied = (1 - disrupted) + disrupted * supervision
    #   overall_displacement = (1 - min(worker_occupied, 1)) * remote_share
    #   adjusted_unemp = base + overall_disp - overall_disp * jevons
    #                   = base + overall_disp * (1 - jevons)
    disrupted = rli_traj_lagged * rli_cov
    del rli_traj_lagged  # free (n_samples, n_days) array
    overall_displacement = np.maximum(disrupted * (1 - supervision), 0) * remote_share
    del disrupted
    adjusted_unemp = base_unemp_arr + overall_displacement * (1 - jevons_samples[:, None])
    adjusted_unemp_pct = adjusted_unemp * 100
    del overall_displacement

    # Percentiles
    pct5 = np.percentile(adjusted_unemp_pct, 5, axis=0)
    pct10 = np.percentile(adjusted_unemp_pct, 10, axis=0)
    pct25 = np.percentile(adjusted_unemp_pct, 25, axis=0)
    pct50 = np.percentile(adjusted_unemp_pct, 50, axis=0)
    pct75 = np.percentile(adjusted_unemp_pct, 75, axis=0)
    pct90 = np.percentile(adjusted_unemp_pct, 90, axis=0)
    pct95 = np.percentile(adjusted_unemp_pct, 95, axis=0)

    # Also compute RLI percentiles for display
    rli_pct_pct = rli_traj_frac * 100
    rli_p50 = np.percentile(rli_pct_pct, 50, axis=0)

    # ── Jobs lost above baseline ────────────────────────────────────────
    _is_jobs_mode = emp_display_mode == "Jobs Lost Above Baseline"
    if _is_jobs_mode:
        # Jobs above baseline = (adjusted_unemp - base_unemp) * labor_force
        jobs_above_baseline = (adjusted_unemp - base_unemp_arr) * emp_labor_force  # millions
        jobs_pct5  = np.percentile(jobs_above_baseline, 5, axis=0)
        jobs_pct10 = np.percentile(jobs_above_baseline, 10, axis=0)
        jobs_pct25 = np.percentile(jobs_above_baseline, 25, axis=0)
        jobs_pct50 = np.percentile(jobs_above_baseline, 50, axis=0)
        jobs_pct75 = np.percentile(jobs_above_baseline, 75, axis=0)
        jobs_pct90 = np.percentile(jobs_above_baseline, 90, axis=0)
        jobs_pct95 = np.percentile(jobs_above_baseline, 95, axis=0)

    # ── Main content ─────────────────────────────────────────────────────
    st.header("AI Employment Displacement Model")

    # ── Fan chart ──────────────────────────────────────────────────────────
    fig = go.Figure()

    if _is_jobs_mode:
        _chart_lo5, _chart_hi95 = jobs_pct5, jobs_pct95
        _chart_lo10, _chart_hi90 = jobs_pct10, jobs_pct90
        _chart_lo25, _chart_hi75 = jobs_pct25, jobs_pct75
        _chart_med = jobs_pct50
    else:
        _chart_lo5, _chart_hi95 = pct5, pct95
        _chart_lo10, _chart_hi90 = pct10, pct90
        _chart_lo25, _chart_hi75 = pct25, pct75
        _chart_med = pct50

    # Fan bands
    bands_spec = [
        (_chart_lo5, _chart_hi95, 'rgba(231,76,60,0.10)', '90% CI'),
        (_chart_lo10, _chart_hi90, 'rgba(231,76,60,0.18)', '80% CI'),
        (_chart_lo25, _chart_hi75, 'rgba(231,76,60,0.28)', '50% CI'),
    ]
    for lo, hi, color, label in bands_spec:
        x_poly = proj_dates + proj_dates[::-1]
        y_poly = list(hi) + list(lo[::-1])
        fig.add_trace(go.Scatter(
            x=x_poly, y=y_poly,
            fill='toself', fillcolor=color,
            line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True,
        ))

    # Median line
    if _is_jobs_mode:
        hover_med = [f"{dt.strftime('%b %d, %Y')}<br>Jobs lost: {_fmt_jobs(y)}<br>RLI: {r:.1f}%"
                     for dt, y, r in zip(proj_dates, _chart_med, rli_p50)]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=_chart_med.tolist(),
            mode='lines', line=dict(color='#e74c3c', width=2.5),
            name='Median jobs lost',
            hovertext=hover_med, hoverinfo='text',
        ))
    else:
        hover_med = [f"{dt.strftime('%b %d, %Y')}<br>Unemployment: {y:.1f}%<br>RLI: {r:.1f}%"
                     for dt, y, r in zip(proj_dates, _chart_med, rli_p50)]
        fig.add_trace(go.Scatter(
            x=proj_dates, y=_chart_med.tolist(),
            mode='lines', line=dict(color='#e74c3c', width=2.5),
            name='Median unemployment',
            hovertext=hover_med, hoverinfo='text',
        ))

    if not _is_jobs_mode:
        # Base unemployment reference line
        fig.add_trace(go.Scatter(
            x=[proj_dates[0], proj_dates[-1]],
            y=[emp_base_unemp, emp_base_unemp],
            mode='lines', line=dict(color='#7f8c8d', width=1.5, dash='dot'),
            name=f'Base unemployment ({emp_base_unemp:.1f}%)',
            hoverinfo='skip',
        ))

    # Milestone lines
    if _is_jobs_mode:
        _jobs_ms = [
            (1,  "1M",  '#888888'),
            (5,  "5M",  '#e67e22'),
            (10, "10M", '#c0392b'),
            (20, "20M", '#8e44ad'),
            (30, "30M", '#2c3e50'),
        ]
        _ms_max = max(_chart_hi95[-1], 1) + 1
        for ms_val, ms_label, ms_color in _jobs_ms:
            if ms_val <= _ms_max:
                fig.add_trace(go.Scatter(
                    x=[proj_dates[0], proj_dates[-1]], y=[ms_val, ms_val],
                    mode='lines', line=dict(color=ms_color, width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ))
                fig.add_annotation(
                    x=1.0, xref='paper', y=ms_val, text=f"  {ms_label}",
                    showarrow=False, xanchor='left', yanchor='middle',
                    font=dict(size=10, color=ms_color))
    else:
        for unemp_val, label, color in [
            (5,  "5%",  '#888888'),
            (10, "10%", '#e67e22'),
            (15, "15%", '#c0392b'),
            (20, "20%", '#8e44ad'),
            (25, "25%", '#2c3e50'),
        ]:
            if unemp_val <= max(_chart_hi95[-1], 10):
                fig.add_trace(go.Scatter(
                    x=[proj_dates[0], proj_dates[-1]], y=[unemp_val, unemp_val],
                    mode='lines', line=dict(color=color, width=1, dash='dot'),
                    hoverinfo='skip', showlegend=False,
                ))
                fig.add_annotation(
                    x=1.0, xref='paper', y=unemp_val, text=f"  {label}",
                    showarrow=False, xanchor='left', yanchor='middle',
                    font=dict(size=10, color=color))

    today = _add_today_vline(fig)

    if _is_jobs_mode:
        y_max = max(_chart_hi95[-1], 1) + 1
        _y_title = "Jobs Lost Above Baseline (millions)"
        _y_suffix = 'M'
    else:
        y_max = max(_chart_hi95[-1], 10) + 2
        _y_title = "Adjusted Unemployment Rate (%)"
        _y_suffix = '%'

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=140, t=50, b=40),
        font=dict(color='#1a1a2e'),
        xaxis=dict(
            range=[proj_dates[0] - timedelta(days=10),
                   proj_end_date + timedelta(days=30)],
            gridcolor='rgba(0,0,0,0.1)',
            tickfont=dict(color='#1a1a2e'),
            zeroline=False,
        ),
        yaxis=dict(
            title=_y_title,
            range=[0, y_max],
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=False,
            ticksuffix=_y_suffix,
            tickfont=dict(color='#1a1a2e'),
            title_font=dict(color='#1a1a2e'),
        ),
        hovermode='x unified',
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                    bgcolor='rgba(255,255,255,0.95)',
                    font=dict(color='#1a1a2e')),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    st.plotly_chart(fig, width="stretch")

    # ── Projections row ──────────────────────────────────────────────────
    eoy_targets = [
        ("Today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026 Jun", datetime(2026, 6, 30)),
        ("2026EOY", datetime(2026, 12, 31)),
        ("2027EOY", datetime(2027, 12, 31)),
        ("2028EOY", datetime(2028, 12, 31)),
        ("2029EOY", datetime(2029, 12, 31)),
    ]

    def _emp_at_date(target_date):
        elapsed = (target_date - emp_current['date']).days
        if elapsed < 0:
            elapsed = 0
        idx = min(elapsed, len(proj_days_arr) - 1)
        return adjusted_unemp_pct[:, idx], rli_pct_pct[:, idx], idx

    valid_targets = [(label, d) for label, d in eoy_targets if d <= proj_end_date]
    if valid_targets:
        cols = st.columns(len(valid_targets))
        for col, (label, target_date) in zip(cols, valid_targets):
            unemp_samples, rli_samples, _t_idx = _emp_at_date(target_date)
            p50_r = np.percentile(rli_samples, 50)
            with col:
                if _is_jobs_mode:
                    _jobs_samples = jobs_above_baseline[:, _t_idx]
                    p10_j, p50_j, p90_j = np.percentile(_jobs_samples, [10, 50, 90])
                    st.metric(label=label, value=_fmt_jobs(p50_j))
                    st.caption(f"80% CI: {_fmt_jobs(p10_j)} – {_fmt_jobs(p90_j)}\nRLI: {p50_r:.1f}%")
                else:
                    p10_u, p50_u, p90_u = np.percentile(unemp_samples, [10, 50, 90])
                    st.metric(label=label, value=f"{p50_u:.1f}%")
                    st.caption(f"80% CI: {p10_u:.1f}% – {p90_u:.1f}%\nRLI: {p50_r:.1f}%")

    # ── Model breakdown at a reference date ──────────────────────────────
    with st.expander("Model breakdown"):
        _bd_options = {label: d for label, d in valid_targets}
        _bd_default_label = "2027EOY" if "2027EOY" in _bd_options else list(_bd_options.keys())[-1]
        _bd_selected = st.selectbox("Reference date", list(_bd_options.keys()),
                                     index=list(_bd_options.keys()).index(_bd_default_label),
                                     key="emp_breakdown_date")
        _ref_date = _bd_options[_bd_selected]
        _ref_elapsed = max((_ref_date - emp_current['date']).days, 0)
        _ref_idx = min(_ref_elapsed, len(proj_days_arr) - 1)
        _ref_rli_raw = np.median(rli_traj_frac[:, _ref_idx]) * 100
        _ref_lag_days = int(emp_lag_days)
        _ref_lag_target_date = _ref_date - timedelta(days=_ref_lag_days)
        if _ref_lag_target_date >= emp_current['date']:
            # Lagged date is within projection range
            _ref_lagged_idx = min((_ref_lag_target_date - emp_current['date']).days, len(proj_days_arr) - 1)
            _ref_rli = np.median(rli_traj_frac[:, _ref_lagged_idx]) * 100
        else:
            # Lagged date falls before projection start — look up historical RLI
            _ref_rli = 0.4  # floor: assume ~0.4% RLI before first indexed model
            for _hist_m in rli_frontier_all:
                if _hist_m['date'] <= _ref_lag_target_date:
                    _ref_rli = _hist_m['rli_score']
                else:
                    break

        _ref_bu_sigma = _bu_sigma_1yr * np.sqrt(_ref_elapsed / 365.0)
        _ref_base_unemp = emp_base_unemp  # median stays at slider value

        _ref_disrupted = _ref_rli / 100 * rli_cov
        _ref_overhead = _ref_disrupted * supervision
        _ref_worker_occ = (1 - _ref_disrupted) + _ref_overhead
        _ref_headcount = min(_ref_worker_occ, 1.0)
        _ref_remote_disp = 1 - _ref_headcount
        _ref_overall_disp = _ref_remote_disp * remote_share
        _ref_raw_unemp = _ref_base_unemp / 100 + _ref_overall_disp
        _ref_adj_unemp = _ref_raw_unemp - _ref_overall_disp * (emp_jevons / 100)

        _lag_months = _ref_lag_days / 30.4
        st.markdown(f"**Breakdown at {_ref_date.strftime('%b %Y')} (raw RLI = {_ref_rli_raw:.1f}%, lagged RLI = {_ref_rli:.1f}%, lag = {_lag_months:.0f}mo)**")

        if _is_jobs_mode:
            _ref_lf = emp_labor_force  # millions
            _ref_remote_jobs = _ref_lf * remote_share
            _ref_displaced_remote = _ref_remote_jobs * _ref_remote_disp
            _ref_overall_jobs = _ref_lf * _ref_overall_disp
            _ref_jevons_recovery = _ref_overall_jobs * (emp_jevons / 100)
            _ref_net_jobs_lost = _ref_overall_jobs - _ref_jevons_recovery
            breakdown_rows = [
                {"Step": "0. Raw RLI Score (before lag)", "Value": f"{_ref_rli_raw:.1f}%"},
                {"Step": f"1. Lagged RLI Score (−{_ref_lag_days}d)", "Value": f"{_ref_rli:.1f}%"},
                {"Step": f"2. Disrupted fraction ({_ref_rli:.1f}% × {rli_cov*100:.0f}% coverage)", "Value": f"{_ref_disrupted*100:.1f}%"},
                {"Step": f"3. Overhead hours ({_ref_disrupted*100:.1f}% × {supervision*100:.0f}% supervision)", "Value": f"{_ref_overhead*100:.1f}%"},
                {"Step": f"4. Worker occupied fraction (1 − {_ref_disrupted*100:.1f}% + {_ref_overhead*100:.1f}%)", "Value": f"{_ref_worker_occ*100:.1f}%"},
                {"Step": "5. Headcount needed (capped at 100%)", "Value": f"{_ref_headcount*100:.1f}%"},
                {"Step": f"6. Remote displacement rate (1 − {_ref_headcount*100:.1f}%)", "Value": f"{_ref_remote_disp*100:.1f}%"},
                {"Step": f"7. Remote/digital jobs ({_ref_lf:.0f}M × {remote_share*100:.0f}% remote share)", "Value": f"{_fmt_jobs(_ref_remote_jobs)}"},
                {"Step": f"8. Remote jobs displaced ({_fmt_jobs(_ref_remote_jobs)} × {_ref_remote_disp*100:.1f}%)", "Value": f"{_fmt_jobs(_ref_displaced_remote)} ({_ref_displaced_remote*1e6:.0f})"},
                {"Step": f"9. Overall jobs displaced (= remote displaced)", "Value": f"{_fmt_jobs(_ref_overall_jobs)} ({_ref_overall_jobs*1e6:.0f})"},
                {"Step": f"10. Jevons/reallocation recovery ({_fmt_jobs(_ref_overall_jobs)} × {emp_jevons:.0f}%)", "Value": f"−{_fmt_jobs(_ref_jevons_recovery)}"},
                {"Step": f"11. Net jobs lost above baseline ({_fmt_jobs(_ref_overall_jobs)} − {_fmt_jobs(_ref_jevons_recovery)})", "Value": f"{_fmt_jobs(_ref_net_jobs_lost)} ({_ref_net_jobs_lost*1e6:.0f})"},
            ]
        else:
            breakdown_rows = [
                {"Step": "0. Raw RLI Score (before lag)", "Value": f"{_ref_rli_raw:.1f}%"},
                {"Step": f"1. Lagged RLI Score (−{_ref_lag_days}d)", "Value": f"{_ref_rli:.1f}%"},
                {"Step": f"2. Disrupted fraction ({_ref_rli:.1f}% × {rli_cov*100:.0f}% coverage)", "Value": f"{_ref_disrupted*100:.1f}%"},
                {"Step": f"3. Overhead hours ({_ref_disrupted*100:.1f}% × {supervision*100:.0f}% supervision)", "Value": f"{_ref_overhead*100:.1f}%"},
                {"Step": f"4. Worker occupied fraction (1 − {_ref_disrupted*100:.1f}% + {_ref_overhead*100:.1f}%)", "Value": f"{_ref_worker_occ*100:.1f}%"},
                {"Step": "5. Headcount needed (capped at 100%)", "Value": f"{_ref_headcount*100:.1f}%"},
                {"Step": f"6. Remote displacement rate (1 − {_ref_headcount*100:.1f}%)", "Value": f"{_ref_remote_disp*100:.1f}%"},
                {"Step": f"7. Overall displacement ({_ref_remote_disp*100:.1f}% × {remote_share*100:.0f}% remote share)", "Value": f"{_ref_overall_disp*100:.1f}%"},
                {"Step": f"8. Raw unemployment ({_ref_base_unemp:.1f}%±{_ref_bu_sigma:.1f}% base + {_ref_overall_disp*100:.1f}% displacement)", "Value": f"{_ref_raw_unemp*100:.1f}%"},
                {"Step": f"9. Adjusted unemployment ({_ref_raw_unemp*100:.1f}% − {_ref_overall_disp*100:.1f}% × {emp_jevons:.0f}% Jevons)", "Value": f"{_ref_adj_unemp*100:.1f}%"},
            ]
        st.table(breakdown_rows)

    # ── Milestone table ──────────────────────────────────────────────────
    if _is_jobs_mode:
        _emp_milestones = [
            (1,  "1M jobs lost"),
            (5,  "5M jobs lost"),
            (10, "10M jobs lost"),
            (20, "20M jobs lost"),
            (30, "30M jobs lost"),
        ]
        _ms_data = jobs_above_baseline  # (n_samples, n_timesteps)
        _ms_unit = "M"
    else:
        _emp_milestones = [
            (5,  "5% unemployment"),
            (8,  "8% unemployment"),
            (10, "10% unemployment"),
            (15, "15% unemployment"),
            (20, "20% unemployment"),
        ]
        _ms_data = adjusted_unemp_pct
        _ms_unit = "%"

    with st.expander("Milestone details"):
        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("**Probabilities**")
            rows = []
            for threshold, ms_label in _emp_milestones:
                row = {"Milestone": ms_label}
                for eoy_label, target_date in valid_targets:
                    _, _, _t_idx = _emp_at_date(target_date)
                    if _is_jobs_mode:
                        _ms_samples = jobs_above_baseline[:, _t_idx]
                    else:
                        _ms_samples, _, _ = _emp_at_date(target_date)
                    prob = np.mean(_ms_samples >= threshold) * 100
                    row[eoy_label] = f"{prob:.0f}%"
                rows.append(row)
            st.table(rows)

        with tcol2:
            st.markdown("**Estimated arrival**")
            arrival_rows = []
            for threshold, ms_label in _emp_milestones:
                crossed = np.argmax(_ms_data >= threshold, axis=1)
                actually_crossed = _ms_data[np.arange(n_samples), crossed] >= threshold
                if actually_crossed.sum() < n_samples * 0.05:
                    arrival_rows.append({"Milestone": ms_label, "Median": "Beyond range", "80% CI": "—"})
                    continue
                crossed_days = crossed[actually_crossed]
                p10_d = int(np.percentile(crossed_days, 10))
                p50_d = int(np.percentile(crossed_days, 50))
                p90_d = int(np.percentile(crossed_days, 90))
                med_date = emp_current['date'] + timedelta(days=p50_d)
                early_date = emp_current['date'] + timedelta(days=p10_d)
                late_date = emp_current['date'] + timedelta(days=p90_d)
                arrival_rows.append({
                    "Milestone": ms_label,
                    "Median": med_date.strftime('%b %Y'),
                    "80% CI": f"{early_date.strftime('%b %Y')} – {late_date.strftime('%b %Y')}",
                })
            st.table(arrival_rows)

    _fn_caption(
        "A simple economic displacement model \u2014 what it assumes, and what "
        "it leaves out. RLI data from remotelabor.ai. These are projections, "
        "not forecasts.",
        ("what it assumes", "AI automation displaces remote/digital work "
                            "proportional to a lagged RLI score, with partial "
                            "recovery from Jevons paradox and worker "
                            "reallocation."),
        ("what it leaves out", "Labor-market dynamics, wage effects, new job "
                               "creation and policy responses."),
        ("projections, not forecasts", PROJ_DISCLAIMER.strip()))


# ── ECI Company Gap ────────────────────────────────────────────────────

# Organization substring -> gap-tab display name, inverted from _ECI_COMPANIES
# so this tab and the Epoch ECI tab resolve companies from one table. Edit the
# registry, not this. Consumed by _ecg_org_display() below.
_ECG_ORG_MAP = {o: n for n, c in _ECI_COMPANIES.items() for o in c["orgs"]}


def _ecg_org_display(org_raw):
    """Map Epoch's raw `Organization` string to a gap-tab display name.

    Substring match, longest key first, returning None when nothing matches.

    This deliberately mirrors the substring semantics `load_eci_frontier(orgs=…)`
    uses for the Epoch ECI tab. The two tabs previously disagreed because this
    one did an exact dict lookup: Epoch spells Google four different ways
    ("Google DeepMind", "Google", "Google DeepMind,Google", "Google,Google
    DeepMind"), so an exact map keyed only on "Google DeepMind" silently dropped
    four Google models and put a different point on Google's 2025 frontier
    (Gemini 2.0 Pro Exp 135.43 on 2025-02-05, where the ECI tab correctly showed
    Gemini 2.0 Flash Thinking Exp 136.00 on 2025-01-21). Matching by substring
    keeps the two tabs consistent by construction, so a new spelling in a future
    Epoch pull cannot desync them again. TestEcgOrgMatching guards this.

    Longest-key-first keeps the result deterministic when one key is a substring
    of another. The registry keeps `orgs` minimal so that rarely applies today,
    but it costs nothing and removes a footgun from adding a variant later. A
    census of every distinct Organization string in the CSV finds none matching
    two *different* companies, so the match is unambiguous; TestEcgOrgMatching
    re-checks that on each run against the live data.
    """
    o = (org_raw or "").lower()
    for k in sorted(_ECG_ORG_MAP, key=len, reverse=True):
        if k.lower() in o:
            return _ECG_ORG_MAP[k]
    return None


# Both derived from _ECI_COMPANIES. (There was also an _ECG_DASH table of
# per-company line styles; nothing in any render path ever read it -- the gap tab
# styles one highlighted company at a time via _ECG_COLORS -- so it was removed
# rather than carried as another table to keep in sync.)
_ECG_COLORS = {n: c["color"] for n, c in _ECI_COMPANIES.items()}
_ECG_COUNTRY = {n: c["country"] for n, c in _ECI_COMPANIES.items()}

_ECG_FLAG = {"US": "\U0001f1fa\U0001f1f8", "CN": "\U0001f1e8\U0001f1f3", "FR": "\U0001f1eb\U0001f1f7"}


def _ecg_frontier_date_at_score(frontier_pts, target_score):
    """Interpolate: when did the overall frontier first reach target_score?
    frontier_pts: list of (datetime, score) sorted by date, scores monotonically increasing.
    Returns datetime or None if target exceeds current frontier.
    """
    if not frontier_pts:
        return None
    if target_score <= frontier_pts[0][1]:
        return frontier_pts[0][0]
    if target_score > frontier_pts[-1][1]:
        return None
    for i in range(1, len(frontier_pts)):
        d0, s0 = frontier_pts[i - 1]
        d1, s1 = frontier_pts[i]
        if s0 <= target_score <= s1:
            if s1 == s0:
                return d0
            frac = (target_score - s0) / (s1 - s0)
            delta = (d1 - d0).total_seconds() * frac
            return d0 + timedelta(seconds=delta)
    return None


def _ecg_frontier_score_at_date(frontier_pts, target_date):
    """Step function: what was the frontier score at target_date?
    frontier_pts: list of (datetime, score) sorted by date, scores monotonically increasing.
    Returns the score of the most recent frontier-setter on or before target_date.
    """
    if not frontier_pts:
        return None
    score = None
    for d, s in frontier_pts:
        if d <= target_date:
            score = s
        else:
            break
    if score is None:
        return frontier_pts[0][1]
    return score


def render_eci_gap():
    # ── Build overall frontier from eci_all ──
    all_models = sorted(eci_all, key=lambda m: m['date'])
    overall_frontier = []  # (date, score) monotonically increasing
    max_score = -float('inf')
    for m in all_models:
        if m['eci_score'] > max_score:
            max_score = m['eci_score']
            overall_frontier.append((m['date'], m['eci_score']))

    # ── Build per-org frontiers ──
    org_models = {}
    for m in all_models:
        org_raw = m.get('organization', '')
        display = _ecg_org_display(org_raw)
        if not display:
            continue
        org_models.setdefault(display, []).append(m)

    org_frontiers = {}
    for org, models in org_models.items():
        best = -float('inf')
        frontier_pts = []
        for m in models:
            if m['eci_score'] > best:
                best = m['eci_score']
                fdate = _ecg_frontier_date_at_score(overall_frontier, m['eci_score'])
                if fdate is not None:
                    gap_months = (m['date'] - fdate).total_seconds() / (30.44 * 86400)
                else:
                    gap_months = 0.0
                frontier_score_at_release = _ecg_frontier_score_at_date(
                    overall_frontier, m['date'])
                frontier_pts.append({
                    'date': m['date'],
                    'score': m['eci_score'],
                    'name': m.get('display_name', m.get('name', '')),
                    'gap_months': max(0.0, gap_months),
                    'frontier_score': frontier_score_at_release,
                })
        org_frontiers[org] = frontier_pts

    # Current frontier score (used to determine who actually owns the frontier)
    _current_frontier_score = overall_frontier[-1][1] if overall_frontier else None

    # ── Gap formatting helper ──
    def _fmt_gap(months, at_frontier=False):
        """Format a gap. 'At frontier' is strict: the org owns the current frontier."""
        if at_frontier:
            return "At frontier"
        if months < 0.5:
            return "<1mo"
        return f"{months:.0f}mo"

    def _fmt_release_gap(months):
        """Gap-at-release formatter. Models that set the frontier show 'At frontier'."""
        if months < 0.1:
            return "At frontier"
        if months < 0.5:
            return "<1mo"
        return f"{months:.0f}mo"

    # ── Compute current effective gap (staleness-adjusted) ──
    _today = datetime.now()
    org_current = {}
    for org, pts in org_frontiers.items():
        if not pts:
            continue
        latest = pts[-1]
        # Strict "at frontier" = currently owns (or ties) the max frontier score
        is_at_frontier = (
            _current_frontier_score is not None
            and latest['score'] >= _current_frontier_score
        )
        if is_at_frontier:
            effective_gap = 0.0
        else:
            fdate = _ecg_frontier_date_at_score(overall_frontier, latest['score'])
            if fdate is not None:
                effective_gap = max(
                    0.0, (_today - fdate).total_seconds() / (30.44 * 86400))
            else:
                effective_gap = 0.0
        model_age_months = (_today - latest['date']).total_seconds() / (30.44 * 86400)
        current_eci_gap = (_current_frontier_score - latest['score']) if _current_frontier_score is not None else 0.0
        org_current[org] = {
            'name': latest['name'],
            'date': latest['date'],
            'score': latest['score'],
            'gap_at_release': latest['gap_months'],
            'effective_gap': effective_gap,
            'model_age_months': max(0.0, model_age_months),
            'at_frontier': is_at_frontier,
            'current_eci_gap': max(0.0, current_eci_gap),
        }

    # Sort orgs: current frontier holders first, then chasers by effective gap
    _frontier_orgs = {o for o, info in org_current.items() if info['at_frontier']}
    _chaser_orgs = [o for o in org_current if o not in _frontier_orgs]
    _chaser_orgs.sort(key=lambda o: org_current[o]['effective_gap'])
    all_orgs = sorted(_frontier_orgs) + _chaser_orgs

    # ── Sidebar ──
    with st.sidebar:
        st.header("ECI Company Gap")
        highlight_org = st.selectbox(
            "Highlight company", ["None"] + all_orgs,
            key="ecg_highlight")
        if highlight_org == "None":
            highlight_org = None

    # ── Header ──
    st.header("ECI Company Gap: Months Behind Frontier")
    frontier_model = overall_frontier[-1] if overall_frontier else None
    if frontier_model:
        frontier_date, frontier_score = frontier_model
        st.markdown(f"**Current frontier:** ECI **{frontier_score:.1f}** "
                    f"(set {frontier_date.strftime('%b %d, %Y')})")

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Selected Company Gap Over Time (shown first, before table)
    # ══════════════════════════════════════════════════════════════════════
    if highlight_org and highlight_org in org_frontiers:
        h_pts = org_frontiers[highlight_org]
        h_flag = _ECG_FLAG.get(_ECG_COUNTRY.get(highlight_org, ""), "")
        h_color = _ECG_COLORS.get(highlight_org, '#888888')
        h_info = org_current[highlight_org]

        st.subheader(f"{h_flag} {highlight_org}: Months Behind Frontier Over Time")
        gap_release = h_info['gap_at_release']
        gap_now = h_info['effective_gap']
        eci_gap_now = h_info['current_eci_gap']
        if h_info['at_frontier']:
            gap_desc = "Currently **at frontier**."
        elif gap_release < 0.5:
            gap_desc = (
                f"Was at frontier at release, now **{gap_now:.1f}mo behind** "
                f"(frontier moved to {_current_frontier_score:.1f}, +{eci_gap_now:.1f} ECI).")
        else:
            gap_desc = (
                f"Was **{gap_release:.1f}mo behind** at release, "
                f"now **{gap_now:.1f}mo behind** "
                f"(+{eci_gap_now:.1f} ECI from current frontier).")
        st.markdown(
            f"Best model: **{h_info['name']}** (ECI {h_info['score']:.1f}, "
            f"{h_info['date'].strftime('%b %Y')}). {gap_desc}")

        fig_h = go.Figure()

        x_start = h_pts[0]['date'] - timedelta(days=30)
        x_end = _today + timedelta(days=30)

        h_dates = [p['date'] for p in h_pts]
        h_gaps = [p['gap_months'] for p in h_pts]
        h_names = [p['name'] for p in h_pts]
        h_hover = []
        for p in h_pts:
            fscore = p.get('frontier_score')
            if fscore is not None:
                eci_gap = fscore - p['score']
                h_hover.append(
                    f"{p['name']}<br>{p['date'].strftime('%b %Y')}<br>"
                    f"Model ECI: {p['score']:.1f}<br>"
                    f"Frontier ECI: {fscore:.1f}<br>"
                    f"ECI gap: {eci_gap:+.1f}<br>"
                    f"Time behind: {p['gap_months']:.1f}mo")
            else:
                h_hover.append(
                    f"{p['name']}<br>{p['date'].strftime('%b %Y')}<br>"
                    f"Model ECI: {p['score']:.1f}<br>"
                    f"Time behind: {p['gap_months']:.1f}mo")

        fig_h.add_trace(go.Scatter(
            x=h_dates, y=h_gaps,
            mode='lines+markers+text',
            marker=dict(color=h_color, size=10, line=dict(color='white', width=1.5)),
            line=dict(color=h_color, width=3),
            text=h_names, textposition='top center',
            textfont=dict(size=9, color=h_color),
            hovertext=h_hover, hoverinfo='text',
            showlegend=False,
        ))

        # Extend line to today whenever the org is not currently at frontier,
        # OR its best model is stale enough that staleness is meaningful.
        _show_today_marker = (not h_info['at_frontier']) or h_info['model_age_months'] > 2
        if _show_today_marker:
            fig_h.add_trace(go.Scatter(
                x=[h_pts[-1]['date'], _today],
                y=[h_gaps[-1], h_info['effective_gap']],
                mode='lines',
                line=dict(color=h_color, width=2, dash='dash'),
                hoverinfo='skip', showlegend=False,
            ))
            _today_frontier = _ecg_frontier_score_at_date(overall_frontier, _today)
            if _today_frontier is not None:
                _today_eci_gap = _today_frontier - h_info['score']
                _today_hover = (
                    f"Today<br>"
                    f"Model ECI: {h_info['score']:.1f}<br>"
                    f"Frontier ECI: {_today_frontier:.1f}<br>"
                    f"ECI gap: {_today_eci_gap:+.1f}<br>"
                    f"Time behind: {h_info['effective_gap']:.1f}mo<br>"
                    f"(no new model in {h_info['model_age_months']:.1f}mo)")
            else:
                _today_hover = (
                    f"Today<br>Effective gap: {h_info['effective_gap']:.1f}mo<br>"
                    f"(no new model in {h_info['model_age_months']:.1f}mo)")
            _today_label = (f"Today: {h_info['effective_gap']:.1f}mo"
                            if h_info['effective_gap'] >= 0.05
                            else "Today: 0mo")
            fig_h.add_trace(go.Scatter(
                x=[_today], y=[h_info['effective_gap']],
                mode='markers+text',
                marker=dict(color=h_color, size=12, symbol='diamond',
                            line=dict(color='white', width=1.5)),
                text=[_today_label],
                textposition='top center',
                textfont=dict(size=10, color=h_color),
                hovertext=_today_hover,
                hoverinfo='text', showlegend=False,
            ))

        fig_h.add_hline(y=0, line=dict(color='black', width=1.5, dash='dot'))

        _y_max = max(h_gaps + [1.0])
        if _show_today_marker:
            _y_max = max(_y_max, h_info['effective_gap'] + 1)

        fig_h.update_layout(
            height=400,
            plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=50, r=50, t=10, b=40),
            font=dict(color='#222222'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.15)',
                       tickfont=dict(color='#222222'), title_font=dict(color='#222222')),
            yaxis=dict(title_text="Months behind frontier",
                       gridcolor='rgba(0,0,0,0.15)',
                       range=[-0.5, _y_max * 1.15],
                       tickfont=dict(color='#222222'), title_font=dict(color='#222222')),
        )
        st.plotly_chart(fig_h, use_container_width=True)
    elif highlight_org is None:
        st.info("Select a company in the sidebar to see its gap vs frontier over time.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Detail Table
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Company Details")

    table_orgs = sorted(org_current.keys(), key=lambda o: org_current[o]['score'], reverse=True)
    rows = []
    for org in table_orgs:
        info = org_current.get(org)
        if not info:
            continue
        flag = _ECG_FLAG.get(_ECG_COUNTRY.get(org, ""), "")
        is_highlighted = (org == highlight_org)
        marker = " **" if is_highlighted else ""
        marker_end = "**" if is_highlighted else ""
        rows.append({
            "Company": f"{marker}{flag} {org}{marker_end}",
            "Best Model": f"{marker}{info['name']}{marker_end}",
            "ECI": f"{marker}{info['score']:.1f}{marker_end}",
            "Released": f"{marker}{info['date'].strftime('%b %Y')}{marker_end}",
            "Age": f"{marker}{info['model_age_months']:.1f}mo{marker_end}",
            "Gap at Release": marker + _fmt_release_gap(info['gap_at_release']) + marker_end,
            "ECI Gap Now": f"{marker}+{info['current_eci_gap']:.1f}{marker_end}",
            "Gap Now": marker + _fmt_gap(info['effective_gap'], info['at_frontier']) + marker_end,
        })

    # Use markdown table for highlighting support
    header = "| Company | Best Model | ECI | Released | Age | Gap at Release | ECI Gap Now | Gap Now |"
    separator = "|---|---|---|---|---|---|---|---|"
    md_rows = [header, separator]
    for r in rows:
        md_rows.append(
            f"| {r['Company']} | {r['Best Model']} | {r['ECI']} | {r['Released']} | "
            f"{r['Age']} | {r['Gap at Release']} | {r['ECI Gap Now']} | {r['Gap Now']} |")
    st.markdown("\n".join(md_rows))



    _fn_caption(
        "Columns: <i>Gap at Release</i>, <i>Gap Now</i>, <i>ECI Gap Now</i>. "
        "Only each org's running-max (best) models are shown. ECI data from "
        "Epoch AI.",
        ("Gap at Release", "Interpolates when the overall ECI frontier first "
                           "reached each model's score level at the time of "
                           "release."),
        ("Gap Now", "How many months ago the frontier first reached the org's "
                    "current best score \u2014 zero iff they still tie or "
                    "exceed the current frontier."),
        ("ECI Gap Now", "The ECI-point gap to the current frontier."))


# ── Data Centers ───────────────────────────────────────────────────────────

_DC_RESET_KEYS = ["dc_metric", "dc_log", "dc_future", "dc_timing", "dc_pool_n",
                  "dc_party", "dc_start_year", "dc_end_year", "dc_cty_cones", "dc_cty_pace",
                  "dc_cty_since"]
_DC_DEFAULTS = {
    "dc_metric": "Compute (H100-equiv)",
    "dc_party": "Tenant (who trains there)",
    "dc_log": True,
    "dc_future": True,
    "dc_timing": "Data center construction",
    "dc_pool_n": "Nearby + announced fabric",
    "dc_start_year": 2025,
    "dc_end_year": 2027,
    "dc_cty_cones": True,
    "dc_cty_pace": "The US trend for every country (a follower tracks the leader)",
    "dc_cty_since": 2024,
}

# Chart/projection window options. The start clips the left edge of every chart
# in the tab; the end caps how far planned buildout is carried forward.
_DC_START_YEARS = [2023, 2024, 2025, 2026]
_DC_END_YEARS = [2026, 2027, 2028, 2029, 2030, 2031]

# What the networked-sites section may pool, ordered weakest-claim first after
# the default. Values name the cluster level fed to _dc_network_site_clusters();
# 'none' pools nothing (each site stands alone) and 'all' pools a company's
# whole fleet regardless of distance, kept only as a stated upper bound.
_DC_NETWORK_OPTIONS = {
    "Nearby + announced fabric": 'fabric',
    "Nearby + plausible fabric": 'plausible',
    "Nearby only": 'proximity',
    "Single site (no networking)": 'none',
    "Every site (implausible)": 'all',
}

# ── All things considered: where the world's compute is ──────────────────
# The Data Centers tab charts shares of what Epoch *catalogues* — frontier
# training sites, overwhelmingly American. Epoch says as much itself: the
# data-center hub covers "a minority of the estimated global AI compute
# stock" (epoch.ai/data/ai-chip-owners, updated May 2026). This section is
# the judgment layer over that catalogue: published country-level estimates
# of the whole installed stock, with their disagreement as the spread.
#
# Anchors, all public and all by *location*, not owner:
#   • Epoch, "Trends in AI supercomputers" (May 2025): US ~75%, China ~15%
#     of aggregate leading-cluster performance; ~74.5% / ~14.1% as re-quoted
#     for 2026. US + China ≈ 90%, which is what pins Other.
#   • Epoch, "Diversion and resale: estimating compute smuggling to China":
#     660k H100e smuggled through 2025 (290k–1.6M) — about a third of
#     China's total compute and about 3% of the global stockpile, i.e.
#     China ≈ 2M H100e of a ≈22M global stock at end-2025, nearer 9%.
#   • AI-2027 tracker: ~12% of AI-relevant compute in China, mid-2026.
#     RAND, as re-quoted: ~15%.
# China's 9–18% band is that disagreement, and it is wider than any single
# source states. SEA and UAE are the weakest numbers here: both regions are
# mostly *pipeline* (Johor ~3 GW approved, Stargate UAE's first 200 MW of a
# planned 1 GW), so today's installed share is small and poorly measured.
_WC_SOURCE_SUPERCOMPUTERS = "https://epoch.ai/data-insights/ai-supercomputers-performance-share-by-country"
_WC_SOURCE_SMUGGLING = "https://epoch.ai/publications/chip-smuggling"
_WC_SOURCE_COVERAGE = "https://epoch.ai/data/ai-chip-owners"
# (label, central share %, 10th pct, 90th pct). Centrals sum to 100; each
# sample is renormalized anyway, so an edited set need not.
_WC_REGIONS = (
    ("US domestic", 72.0, 63.0, 79.0),
    ("China domestic", 13.0, 9.0, 18.0),
    ("SEA", 3.0, 1.5, 6.0),
    ("UAE", 2.0, 0.8, 4.0),
    ("Europe/UK", 6.0, 3.5, 9.0),
    ("Other", 4.0, 2.0, 7.0),
)
_WC_LABELS = tuple(lab for lab, *_ in _WC_REGIONS)
_WC_NOTES = {
    "US domestic":
        "Epoch's leading-cluster share is about 75% (May 2025, ~74.5% "
        "re-quoted for 2026). Set slightly below it, since that series "
        "measures the biggest clusters, where US firms are most "
        "over-represented.",
    "China domestic":
        "The widest band, because the sources disagree most here: Epoch's "
        "cluster share says 14\u201315%, the AI-2027 tracker 12%, and "
        "Epoch's own smuggling estimate implies about 9%. Domestic Ascend "
        "and Cambricon output is the part nobody measures well.",
    "SEA":
        "Johor has roughly 3 GW approved or building, but most of that is "
        "still pipeline, and its tenants are a mix of US hyperscalers and "
        "Chinese labs.",
    "UAE":
        "Stargate UAE's first 200 MW (about 100k GB300s) lands in 2026, "
        "against a planned 1 GW and an announced 5 GW campus.",
    "Europe/UK":
        "The largest piece of what is left after the US and China, and much "
        "of it is US firms building abroad \u2014 Microsoft and Nscale in "
        "Norway and Portugal, Google in the UK \u2014 since these shares go "
        "by location, not owner.",
    "Other":
        "Japan, Korea and India, plus the Gulf outside the UAE \u2014 Saudi "
        "Arabia's Humain brought twin 100 MW campuses online in 2026 and is "
        "aiming at 1.9 GW by 2030. Sized as the remainder: Epoch puts the US "
        "and China together at about 90%.",
}
# Growth of each region's installed compute, ×/yr as (central, p10, p90).
# Only the ratios matter — shares are renormalized at every date.
#   • US: Epoch has leading-cluster performance doubling every ~9 months
#     (~2.5×/yr); the catalogue's US sites aggregate to ~1.9×/yr to end-2028.
#   • China: the catalogue's domestic largest-site fit is ~1.8×/yr, and this
#     tab's own export-control band (_CC_CN_COMPUTE_LO/HI) is 1.4–2.0×/yr for
#     a single cluster.
#   • SEA and the UAE are the fast ones, both from small bases and both mostly
#     announced rather than built — hence the wider bands. The UAE's top end is
#     held at 5×/yr rather than the 6× a from-near-zero lognormal wants: at 6×
#     it compounds to a quarter of world compute by 2029, which the announced
#     5 GW campus does not support even if every phase lands early.
#   • Europe keeps pace with the US because most of its buildout *is* US firms
#     (Microsoft/Nscale, Google, CoreWeave); the catalogue's European sites
#     grow ~1.8×/yr to end-2028.
#   • Other is mature Japan/Korea/India plus a fast Gulf ramp outside the UAE
#     (Saudi Arabia's Humain), hence a middling rate with a wide top.
# How much of each region's growth rate rides on one global shock (chip
# supply, a capex cycle) rather than its own circumstances — the loading on
# that shock, so two regions' log-rates correlate at the product of theirs.
# Drawn independently, the model says the plausible low-US worlds are ones
# where somebody else ran away with the buildout, which has it backwards:
# most non-US capacity is US firms building abroad on US-supplied chips, so
# a US slowdown is a slowdown in SEA, Europe and the Gulf too. China, on
# domestic silicon with export controls already binding, is the one region
# that decouples — and so the one whose share actually rises when the US
# stalls. Loadings leave every region's marginal rate spread exactly as
# stated; only the joint moves.
_WC_COMMON_LOAD = {
    "US domestic": 0.90,
    "Europe/UK": 0.85,
    "SEA": 0.80,
    "UAE": 0.80,
    "Other": 0.70,
    "China domestic": 0.25,
}
_WC_GROWTH = (
    ("US domestic", 2.3, 1.8, 3.0),
    ("China domestic", 1.9, 1.4, 2.6),
    ("SEA", 2.6, 1.8, 4.0),
    ("UAE", 3.5, 2.4, 5.0),
    ("Europe/UK", 2.3, 1.7, 3.2),
    ("Other", 2.2, 1.6, 3.2),
)

# Compute/capabilities/diffusion tab
_CC_RESET_KEYS = ["cc_future", "cc_run", "cc_end_year", "cc_bd_anchor",
                  "cc_company"]
_CC_DEFAULTS = {"cc_future": True, "cc_run": "2-month run",
                "cc_end_year": 2029, "cc_bd_anchor": "Today",
                "cc_company": "OpenAI"}

# Where this tab's time charts open, matching the Data Centers tab's default.
_CC_X_START = datetime(2025, 1, 1)
_CC_BD_OPTIONS = ("Today", "Jan 2025 (backtest)")
_CC_END_YEARS = [2027, 2028, 2029, 2030, 2031]

# Fixed segment boundaries for the frontier-compute growth breakdown. Each entry
# is (label, start, end); the last segment is the planned/under-construction tail.
_CC_SEGMENTS = [
    ("2021–2023", datetime(2021, 1, 1), datetime(2023, 7, 1)),
    ("2023 H2 – 2025 H1", datetime(2023, 7, 1), datetime(2025, 7, 1)),
    ("2025 H2 – today", datetime(2025, 7, 1), None),       # end filled with _today
    ("Planned 2026–2028", None, datetime(2029, 1, 1)),      # start filled with _today
]


def _dc_visible_vals(step_items, x_start):
    """Values a step series shows within [x_start, …]: points at/after x_start
    plus the forward-filled value carried in at the left edge."""
    out = []
    prior = None
    for item in step_items:
        d, v = item[0], item[1]
        if d >= x_start:
            out.append(v)
        elif v is not None:
            prior = v
    if prior is not None:
        out.append(prior)
    return out


def _dc_split_at(items, today, end_x):
    """Split a step series into actual (≤today) and projected (>today) polylines.

    `items` are tuples with [0]=date, [1]=value. Returns (actual_xy, proj_xy) as
    ([x…], [y…]) pairs. The two segments share a boundary point at `today` (the
    forward-filled value) so the solid and dashed lines meet, and each is flattened
    out to its right edge. proj_xy is (None, None) when there's nothing past today.
    """
    v_today = None
    fut = []
    for it in items:
        d, v = it[0], it[1]
        if d <= today:
            v_today = v
        else:
            fut.append((d, v))
    act_x = [it[0] for it in items if it[0] <= today]
    act_y = [it[1] for it in items if it[0] <= today]
    if not fut:
        # No projection; extend actual to the right edge.
        if act_x:
            act_x = act_x + [end_x]
            act_y = act_y + [act_y[-1]]
        return (act_x, act_y), (None, None)
    # Cap actual at today, then run the dashed projection from today onward.
    if v_today is not None:
        act_x = act_x + [today]
        act_y = act_y + [v_today]
        proj_x = [today] + [d for d, _ in fut] + [end_x]
        proj_y = [v_today] + [v for _, v in fut] + [fut[-1][1]]
    else:
        # All points are in the future (rare): dash everything.
        proj_x = [d for d, _ in fut] + [end_x]
        proj_y = [v for _, v in fut] + [fut[-1][1]]
    return (act_x, act_y), (proj_x, proj_y)


def _dc_yrange(values, log_scale):
    """Tight y-axis range from plotted values (log range in log10 units)."""
    vals = [v for v in values if v is not None and (v > 0 if log_scale else True)]
    if not vals:
        return None
    vmin, vmax = min(vals), max(vals)
    if log_scale:
        lo = float(np.floor(np.log10(max(vmin, 1e-9))))
        hi = float(np.ceil(np.log10(max(vmax, 10.0))))
        if hi <= lo:
            hi = lo + 1
        return [lo, hi]
    return [0, vmax * 1.1]


def _dc_tick_label(v, kind):
    """A tick's text, in the same units the value itself is reported in.

    Counts wide enough to need a suffix get one (`_dc_fmt_value`'s k/M), but
    trimmed for an axis: "2M", not "2.00M". Every other kind prints plainly.
    Never rescale a tick against its axis title instead — the H100 axis used to
    divide its labels by a million while the bar text, hovers and the quarterly
    table stayed raw, so a bar reading "1.11M" sat under a "1" tick.
    """
    if kind == 'h100':
        if abs(v) >= 1e6:
            return f"{v / 1e6:g}M"
        if abs(v) >= 1e3:
            return f"{v / 1e3:g}k"
    return f"{v:g}"


def _dc_log_ticks(y_range, kind=None):
    """Explicit log-axis ticks that label every minor tick with its full value
    (e.g. 20, 30, … in the 10-100 decade rather than Plotly's default 2, 3, …),
    while keeping the powers of ten larger so the decade hierarchy stays clear.
    """
    lo = int(np.floor(y_range[0]))
    hi = int(np.ceil(y_range[1]))
    vmin = 10.0 ** y_range[0]
    vmax = 10.0 ** y_range[1]
    vals, text = [], []
    for k in range(lo, hi + 1):
        base = 10.0 ** k
        for m in range(1, 10):
            v = m * base
            if v < vmin * 0.999 or v > vmax * 1.001:
                continue
            label = _dc_tick_label(v, kind)
            if m == 1:
                text.append(label)  # decade label at the axis tickfont size
            else:
                text.append(f"<span style=\"font-size:9px\">{label}</span>")
            vals.append(v)
    return vals, text


def _dc_linear_ticks(y_range, kind=None):
    """Round linear-axis ticks with compact labels, for the kinds whose raw
    counts are too wide to print under a linear axis (H100-equivalents).

    Returns None for every other kind, leaving Plotly's own ticks alone.
    """
    if kind != 'h100' or not y_range:
        return None
    lo, hi = y_range
    span = hi - lo
    if span <= 0:
        return None
    raw = span / 6.0
    mag = 10.0 ** float(np.floor(np.log10(raw)))
    step = next((m * mag for m in (1, 2, 2.5, 5) if m * mag >= raw), 10 * mag)
    vals, text = [], []
    i = int(np.ceil(lo / step))
    while i * step <= hi * (1 + 1e-9):
        v = i * step
        vals.append(v)
        text.append(_dc_tick_label(v, kind))
        i += 1
    return (vals, text) if len(vals) >= 2 else None


# Round durations (in days) used as axis ticks for the 'traintime' metrics.
_DC_DURATION_TICK_DAYS = [
    1 / 24, 2 / 24, 6 / 24, 12 / 24,          # 1h, 2h, 6h, 12h
    1, 2, 4, 7, 14, 30.4375, 60, 91.3125,     # 1d … 3mo
    182.625, 365.25, 730.5, 1826.25,          # 6mo, 1y, 2y, 5y
]


def _dc_duration_ticks(y_range, log_scale):
    """Axis ticks labelled as durations for the 'traintime' metrics.

    The plotted number is runs-per-2mo, so the tick *text* has to be converted
    the same way `_dc_fmt_value` converts a value. Fewer ticks per decade than
    `_dc_log_ticks` because duration labels are much wider than bare numbers.
    """
    if y_range is None:
        return None
    vmin, vmax = ((10.0 ** y_range[0], 10.0 ** y_range[1]) if log_scale
                  else (max(y_range[0], 0.0), y_range[1]))
    # Tick at round durations (1 hour, 1 day, 1 week, …) rather than round
    # run-counts, so the labels read cleanly.
    vals = sorted(_DAYS_2MO / d for d in _DC_DURATION_TICK_DAYS)
    vals = [v for v in vals if vmin * 0.999 <= v <= vmax * 1.001]
    if not vals:
        return None
    return vals, [_dc_fmt_value(v, 'traintime') for v in vals]


def _dc_logop_ticks(y_range, log_scale):
    """Axis ticks labelled in log₁₀ operations for the 'flop' metrics.

    The plotted value is the raw operation count (so every aggregation in the
    tab stays a plain max/sum); only the tick *text* is converted, the same way
    `_dc_fmt_value` converts a value. On a log axis the ticks sit at round log
    values (…, 27.5, 28.0, …) rather than at decade minors, which would round
    to duplicate labels near the top of each decade.
    """
    if y_range is None:
        return None
    if not log_scale:
        # A linear axis is evenly spaced in raw ops, so the ticks are too and
        # only their labels are logged; consecutive labels that round the same
        # are dropped rather than printed twice.
        hi = float(y_range[1])
        if hi <= 0:
            return None
        lo = max(float(y_range[0]), 0.0)
        vals, text = [], []
        for i in range(7):
            v = lo + (hi - lo) * i / 6.0
            lab = _log_op(v)
            if v <= 0 or (text and lab == text[-1]):
                continue
            vals.append(v)
            text.append(lab)
        return (vals, text) if len(vals) >= 2 else None

    lo, hi = float(y_range[0]), float(y_range[1])
    span = hi - lo
    if not np.isfinite(span) or span <= 0:
        return None
    # 0.25 is skipped: a tick at 27.25 would print as "27.2", labelling
    # itself a notch below where it sits.
    step = next((c for c in (0.1, 0.2, 0.5, 1.0, 2.0) if span / c <= 8), 5.0)
    vals, text = [], []
    k = int(np.ceil(lo / step - 1e-9))
    while k * step <= hi + 1e-9:
        t = k * step
        v = 10.0 ** t
        lab = _log_op(v)
        # Whole decades keep the axis tickfont; the steps in between are
        # shrunk, matching the plain log-tick treatment.
        if abs(t - round(t)) > 1e-9:
            lab = f'<span style="font-size:9px">{lab}</span>'
        vals.append(v)
        text.append(lab)
        k += 1
    return (vals, text) if len(vals) >= 2 else None


def _dc_axis_ticks(y_range, log_scale, kind):
    """Tick positions (raw values) and labels for a capacity axis.

    One dispatcher, because the snapshot bar chart builds its own axis rather
    than going through `_dc_layout`: when the two branched separately the bars
    ended up under a differently-labelled axis than the chart above them.
    """
    if kind == 'traintime':
        ticks = _dc_duration_ticks(y_range, log_scale)
    elif kind == 'flop':
        ticks = _dc_logop_ticks(y_range, log_scale)
    elif log_scale and y_range is not None:
        ticks = _dc_log_ticks(y_range, kind)
    else:
        ticks = _dc_linear_ticks(y_range, kind)
    return ticks if ticks and ticks[0] else None


def _dc_layout(log_scale, y_title, x_start, x_end, y_range=None,
               height=440, show_legend=False, kind=None):
    yaxis = dict(title_text=y_title,
                 type='log' if log_scale else 'linear',
                 range=y_range,
                 gridcolor='rgba(0,0,0,0.12)',
                 tickfont=dict(color='#222222'), title_font=dict(color='#222222'))
    ticks = _dc_axis_ticks(y_range, log_scale, kind)
    if ticks is not None:
        yaxis.update(tickmode='array', tickvals=ticks[0], ticktext=ticks[1])
    return dict(
        height=height,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=30, t=10, b=40),
        font=dict(color='#222222'),
        showlegend=show_legend,
        legend=dict(font=dict(size=11, color='#222222'), groupclick='togglegroup'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)', range=[x_start, x_end],
                   tickfont=dict(color='#222222'), title_font=dict(color='#222222')),
        yaxis=yaxis,
    )


def _dc_add_projection_band(fig, today, x_end):
    """Shade the post-today region and draw a labelled 'Today' divider."""
    fig.add_vrect(x0=today, x1=x_end, fillcolor='rgba(120,120,120,0.08)',
                  line_width=0, layer='below')
    fig.add_vline(x=today, line=dict(color='#999999', width=1.5, dash='dot'))
    # Position labels in paper coordinates so they sit at the top of the plot.
    fig.add_annotation(x=today, yref='paper', y=1.0, text='Today',
                       showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(size=10, color='#777777'))
    fig.add_annotation(x=x_end, yref='paper', y=1.0,
                       text='planned / under construction →',
                       showarrow=False, xanchor='right', yanchor='bottom',
                       font=dict(size=10, color='#999999'))


def _dc_fan_bands(fig, grid, traj, color, name, legendgroup):
    """50% and 80% bands of a trajectory matrix, NaN columns left blank."""
    cols = np.where(np.isnan(traj).all(axis=0), np.nan, 0)
    p10, p25, p75, p90 = (np.nanpercentile(traj, q, axis=0) for q in (10, 25, 75, 90))
    for lo, hi, alpha in ((p10, p90, 0.12), (p25, p75, 0.22)):
        rgba = f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},{alpha})"
        xs = [d for d, c in zip(grid, cols) if not np.isnan(c)]
        lo_v = [v for v, c in zip(lo, cols) if not np.isnan(c)]
        hi_v = [v for v, c in zip(hi, cols) if not np.isnan(c)]
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], y=hi_v + lo_v[::-1], fill='toself',
            fillcolor=rgba, line=dict(width=0), mode='lines',
            hoverinfo='skip', showlegend=False, legendgroup=legendgroup,
            name=name))


def _dc_cty_band(arr, kind):
    """'median (p10–p90)' in the metric's units; just the value where the
    samples agree (inside recorded data)."""
    lo, med, hi = (float(np.nanpercentile(arr, q)) for q in (10, 50, 90))
    if lo == hi:
        return _dc_fmt_value(med, kind)
    return (f"{_dc_fmt_value(med, kind)} ({_dc_fmt_value(lo, kind)}–"
            f"{_dc_fmt_value(hi, kind)})")


def _dc_render_country_panel(series, country_of, cluster_of, *, dcs, today, cap_date,
                             x_start, metric_label, kind, log_scale,
                             shift_days, include_future, pace_mode, since,
                             horizon, show_cones=True, run_days=None):
    """Buildout by country — US vs China, with each country's largest training
    run extrapolated past the end of its recorded data under a cone.

    `series` is the tab's metric series with **no** host hidden: a country's
    capacity is a fact about buildings, not about who Epoch lists in them.
    """
    st.subheader("Buildout by country: US vs China")
    st.caption(
        "Biggest training run one party per country could mount. Solid = actual, "
        "dashed = planned, dotted + cone = extrapolated (50% / 80%). "
        "*China-accessible* adds DayOne Johor (FT: Alibaba, ByteDance train there).")

    # Pooling follows the sidebar's networking selector, exactly as the
    # company chart above does: a lone site when nothing pools, else the
    # largest networkable group per company.
    mode = 'site' if cluster_of == {} else 'company'
    # China is drawn twice: the mainland alone, and with the Chinese labs'
    # sites abroad (_DC_CN_ACCESS_ABROAD). Both are projected and tabulated.
    cn_scope = 'abroad'
    cn_key = _DC_CTY_CN_ACCESS

    groups = _dc_country_groups(series, country_of, cn_scope)
    steps_by = {c: _dc_country_steps(series, names, mode, cluster_of)
                for c, names in groups.items()}
    dom = [n for n in series if country_of.get(n) == _DC_CTY_CN]
    steps_by[_DC_CTY_CN_DOMESTIC] = _dc_country_steps(series, dom, mode,
                                                      cluster_of)
    steps_by = {c: s for c, s in steps_by.items()
                if s and any(v and v > 0 for _, v, _ in s)}
    if _DC_CTY_US not in steps_by:
        st.warning("No US data for this metric.")
        return
    horizon_end = datetime(horizon, 12, 31)
    x_end = max(cap_date, horizon_end) + timedelta(days=30)
    grid = _dc_cty_month_grid(x_start, max(cap_date, horizon_end))

    plan_end = today + timedelta(days=_DC_CTY_PLAN_HORIZON_DAYS)
    fits = {c: _dc_cty_fit(s, since=since, t_end=plan_end)
            for c, s in steps_by.items()}
    us_fit = fits[_DC_CTY_US]
    borrowed = set()

    def _pace_for(c):
        """The pace to extrapolate on. A borrowed US pace keeps the country's
        own fitted pace inside the 80% cone, so the two readings disagreeing
        shows up as width rather than vanishing."""
        if c == _DC_CTY_US or (pace_mode == 'own' and fits[c] is not None):
            return fits[c]
        borrowed.add(c)
        if us_fit is None:
            return None
        own = fits[c]
        if own is None:
            return us_fit
        return dict(us_fit, sigma_g=max(us_fit['sigma_g'],
                                        abs(own['g'] - us_fit['g']) / 1.282))

    def _anchor(c):
        """The fit to anchor on — the country's own, or the US pace re-anchored
        at the country's last recorded step when its own history is too short."""
        if fits[c] is not None:
            return fits[c]
        s = steps_by[c]
        if us_fit is None or not s:
            return None
        t0 = min(s[-1][0], plan_end)
        v0 = _dc_val_at([(x[0], x[1]) for x in s], t0)
        if not v0 or v0 <= 0:
            return None
        return dict(us_fit, t0=t0, v0=v0)

    cone_for = [_DC_CTY_US] + [c for c in (cn_key, _DC_CTY_CN_DOMESTIC)
                               if c in steps_by]
    traj, quality = {}, {}
    site_names = dict(groups)
    site_names[_DC_CTY_CN_DOMESTIC] = dom
    for c in cone_for:
        fit = _anchor(c)
        quality[c] = _dc_plan_quality(dcs, site_names.get(c, ()), today)
        traj[c] = _dc_cty_trajectories(
            steps_by[c], fit, grid, N_SAMPLES, pace=_pace_for(c) if fit else None,
            today=today, slip_sigma=_dc_cty_slip_sigma(quality[c]))

    # ── Chart ──
    fig = go.Figure()
    if include_future:
        _dc_add_projection_band(fig, today, x_end)
    others = sorted((c for c in steps_by if c not in cone_for),
                    key=lambda c: -max(v for _, v, _ in steps_by[c] if v))
    for i, c in enumerate(others):
        color = _DC_CTY_OTHER_COLORS[i % len(_DC_CTY_OTHER_COLORS)]
        s = steps_by[c]
        (a_x, a_y), (p_x, p_y) = _dc_split_at(s, today, cap_date)
        fig.add_trace(go.Scatter(
            x=a_x, y=a_y, mode='lines', name=c, legendgroup=c,
            line=dict(color=color, width=1.2, shape='hv'), opacity=0.7,
            hoverinfo='skip'))
        if p_x is not None:
            fig.add_trace(go.Scatter(
                x=p_x, y=p_y, mode='lines', name=c, legendgroup=c,
                line=dict(color=color, width=1.2, shape='hv', dash='dash'),
                opacity=0.7, showlegend=False, hoverinfo='skip'))
        dots = [x for j, x in enumerate(s) if j == 0 or x[1] != s[j - 1][1]]
        fig.add_trace(go.Scatter(
            x=[x[0] for x in dots], y=[x[1] for x in dots], mode='markers',
            marker=dict(size=4, color=color), name=c, legendgroup=c,
            showlegend=False, hoverinfo='text',
            hovertext=[f"{c}{' (planned)' if x[0] > today else ''}<br>"
                       f"{_dc_fmt_value(x[1], kind)} — {x[2]}<br>"
                       f"{_dc_milestone_dates(x[0], shift_days, run_days)}" for x in dots]))
    for c in cone_for:
        color = _DC_CTY_COLORS.get(c, "#D62728")
        s = steps_by[c]
        fit = _anchor(c)
        t0 = fit['t0'] if fit else s[-1][0]
        if show_cones and (t0 > today or (fit is not None and t0 < horizon_end)):
            # The cone opens at today: time-shifted plans up to t0, trend beyond.
            _dc_fan_bands(fig, grid, traj[c], color, c, c)
        if fit is not None and t0 < horizon_end:
            # The dotted median runs from the anchor on; inside the plan
            # window the dashed plan is the reference and the cone hangs off it.
            med = np.nanmedian(traj[c], axis=0)
            start = max(j for j, d in enumerate(grid) if d <= t0)
            proj_cols = [j for j, d in enumerate(grid) if j >= start]
            fig.add_trace(go.Scatter(
                x=[grid[j] for j in proj_cols],
                y=[med[j] for j in proj_cols], mode='lines',
                line=dict(color=color, width=2, dash='dot'), name=c,
                legendgroup=c, showlegend=False, hoverinfo='text',
                hovertext=[f"{c} — {'trend (plan as floor)' if grid[j] <= s[-1][0] else 'extrapolated'}"
                           f"<br>median "
                           f"{_dc_fmt_value(med[j], kind)}<br>80%: "
                           f"{_dc_fmt_value(np.nanpercentile(traj[c][:, j], 10), kind)}"
                           f" – {_dc_fmt_value(np.nanpercentile(traj[c][:, j], 90), kind)}"
                           f"<br>{grid[j]:%b %Y}" for j in proj_cols]))
        # The dashed plan is drawn to its own last catalogued step — passing
        # t0 here made the polyline double back when the catalogue ran past
        # the trend anchor.
        (a_x, a_y), (p_x, p_y) = _dc_split_at(s, today, s[-1][0])
        fig.add_trace(go.Scatter(
            x=a_x, y=a_y, mode='lines', name=c, legendgroup=c,
            line=dict(color=color, width=3, shape='hv'), hoverinfo='skip'))
        if p_x is not None:
            fig.add_trace(go.Scatter(
                x=p_x, y=p_y, mode='lines', name=c, legendgroup=c,
                line=dict(color=color, width=3, shape='hv', dash='dash'),
                showlegend=False, hoverinfo='skip'))
        dots = [x for j, x in enumerate(s) if j == 0 or x[1] != s[j - 1][1]]
        fig.add_trace(go.Scatter(
            x=[x[0] for x in dots], y=[x[1] for x in dots], mode='markers',
            marker=dict(size=6, color=color), name=c, legendgroup=c,
            showlegend=False, hoverinfo='text',
            hovertext=[f"{c}{' (planned)' if x[0] > today else ''}<br>"
                       f"{_dc_fmt_value(x[1], kind)} — {x[2]}<br>"
                       f"{_dc_milestone_dates(x[0], shift_days, run_days)}" for x in dots]))
    vals = [v for s in steps_by.values() for v in _dc_visible_vals(s, x_start)]
    for c in cone_for:
        if not np.isnan(traj[c]).all():
            vals.append(float(np.nanmax(np.nanpercentile(traj[c], 90, axis=0))))
    fig.update_layout(**_dc_layout(log_scale, metric_label, x_start, x_end,
                                   y_range=_dc_yrange(vals, log_scale),
                                   height=500, show_legend=True, kind=kind))
    st.plotly_chart(fig, use_container_width=True)

    # ── Fit readout ──
    def _pace_text(c):
        fit = fits.get(c)
        p = _pace_for(c)
        if p is None:
            return f"{c}: no trend to extrapolate"
        dbl = 12 * np.log10(2) / p['g'] if p['g'] > 0 else float('inf')
        src = ("US pace, cone widened to its own fit" if c in borrowed and fit
               else "US pace" if c in borrowed else "own fit")
        win = ", ".join(f"'{y % 100}: ×{10 ** g:.1f}" for y, g in
                        sorted(fit['windows'].items())) if fit else ""
        q = quality.get(c)
        plan = (f"; plans {q:.0%} sourced, timing "
                f"±{_dc_cty_slip_sigma(q):.0%} of lead (1σ)" if q is not None
                else "; no plans past today")
        return (f"**{c}** ×{10 ** p['g']:.1f}/yr ±{p['sigma_g']:.2f} OOM/yr, "
                f"{src}{'; ' + win if win else ''}{plan}")
    def _pace_short(c):
        p = _pace_for(c)
        if p is None:
            return f"**{c}** no trend"
        return (f"**{c}** ×{10 ** p['g']:.1f}/yr"
                f"{' (US pace)' if c in borrowed else ''}")
    _fn_caption(
        "Pace \u2014 " + "; ".join(_pace_short(c) for c in cone_for)
        + ". Bands centre on the plan.",
        ("Bands centre on the plan", "They widen the further out and the less "
                                     "sourced the plan is; past "
                                     f"{_DC_CTY_PLAN_HORIZON_DAYS // 30} months "
                                     "the trend takes over with plans as a "
                                     "floor."))
    with st.expander("Fit details"):
        _fn_caption(
            "; ".join(_pace_text(c) for c in cone_for)
            + ". Chinese fits rest on a handful of sites.",
            ("OOM/yr", "The \u00b1 is 1\u03c3 on the pace = max(fit s.e., spread "
                       f"across windows, {_DC_CTY_SIGMA_G_FLOOR:.2f}). Timing "
                       "\u03c3 is a fraction of lead time, interpolated by the "
                       "share of future rows whose status cites a document."),
            ("Chinese fits rest on a handful of sites", "And the cones follow "
                                                        "Epoch's catalogue, not "
                                                        "policy."))

    # ── Readout: year-end values, ratio and lag ──
    if cn_key not in traj:
        st.info("No Chinese site has a value for this metric.")
        return
    us, cn = traj[_DC_CTY_US], traj[cn_key]
    dom = traj.get(_DC_CTY_CN_DOMESTIC)
    lag, unresolved = _dc_cty_lag_months(cn, us, grid)
    rows = []
    first_year = max(today.year, grid[0].year)
    for yr in range(first_year, horizon + 1):
        d = datetime(yr, 12, 1)
        if d not in grid:
            continue
        j = grid.index(d)
        if np.isnan(us[:, j]).all() or np.isnan(cn[:, j]).all():
            continue
        u, c = us[:, j], cn[:, j]
        ratio = u / c
        lg = lag[:, j]
        def _num_band(a, fmt):
            lo, med, hi = (float(np.nanpercentile(a, q)) for q in (10, 50, 90))
            return fmt(med) if round(lo) == round(hi) else (
                f"{fmt(med)} ({fmt(lo)}–{fmt(hi)})")
        rows.append({
            "Year end": str(yr),
            "US": _dc_cty_band(u, kind),
            cn_key: _dc_cty_band(c, kind),
            **({_DC_CTY_CN_DOMESTIC: "—" if np.isnan(dom[:, j]).all()
                else _dc_cty_band(dom[:, j], kind)} if dom is not None else {}),
            "US ÷ China": _num_band(ratio, lambda x: f"{x:.1f}×"),
            "China lag (months)": (
                "—" if np.isnan(lg).all() else
                f"ahead in {unresolved[:, j].mean():.0%} of samples"
                if unresolved[:, j].mean() > 0.5 else
                _num_band(lg, lambda x: f"{x:.0f}")),
        })
    if rows:
        last = rows[-1]
        j = grid.index(datetime(horizon, 12, 1)) if datetime(horizon, 12, 1) in grid else None
        if j is not None and not np.isnan(cn[:, j]).all():
            lag_med = float(np.nanmedian(lag[:, j]))
            ahead = float(unresolved[:, j].mean())
            lag_phrase = (
                f"ahead of the US in {ahead:.0%} of samples" if ahead > 0.5 else
                f"about {lag_med:.0f} months behind where the US first stood "
                "at that level" if lag_med >= 0 else
                f"about {-lag_med:.0f} months ahead of the US")
            _fn_line(
                f"**Largest {cn_key} training run by end-{horizon}: "
                f"{_dc_fmt_value(np.nanmedian(cn[:, j]), kind)}** against a US "
                f"{last['US'].split(' (')[0]} — the US at "
                f"{last['US ÷ China'].split(' (')[0]} China's size, {lag_phrase}."
                + (" Mainland China alone is smaller again."
                   if dom is not None and not np.isnan(dom[:, j]).all() else ""),
                (f"{_dc_fmt_value(np.nanmedian(cn[:, j]), kind)}",
                 "80% range: "
                 f"{_dc_fmt_value(np.nanpercentile(cn[:, j], 10), kind)}\u2013"
                 f"{_dc_fmt_value(np.nanpercentile(cn[:, j], 90), kind)}."),
                *([("Mainland China alone",
                    f"{_dc_fmt_value(np.nanmedian(dom[:, j]), kind)} (80%: "
                    f"{_dc_fmt_value(np.nanpercentile(dom[:, j], 10), kind)}\u2013"
                    f"{_dc_fmt_value(np.nanpercentile(dom[:, j], 90), kind)}).")]
                  if dom is not None and not np.isnan(dom[:, j]).all() else []))
        st.table(rows)
        st.caption(
            "Median (10th–90th pct). *Lag* = months since the US first reached "
            "China's value; negative = China ahead.")


def _dc_render_region_share(series, country_of, *, today, cap_date, x_start,
                            metric_label, kind, include_future):
    """Share of all catalogued capacity by where the building stands.

    `series` is the tab's metric series with **no** host hidden and no tenant
    attribution, like the country panel above it: this is a question about
    buildings. Shares are of the total Epoch lists, which is far denser for
    the US than anywhere else — the caption has to keep saying so.
    """
    st.subheader("Share of catalogued capacity by region")

    grid = _dc_cty_month_grid(x_start, cap_date)
    totals = _dc_region_totals(series, country_of, grid)
    grid, shares, levels = _dc_region_shares(totals, grid)
    if not shares:
        st.warning("No capacity in range for this metric.")
        return

    counts = {}
    for name in series:
        lab = _dc_region_of(country_of.get(name, ''))
        counts[lab] = counts.get(lab, 0) + 1
    us_n = counts.get(_DC_REGIONS[0][0], 0)
    rest = sum(counts.values()) - us_n
    # Coverage bias is what the whole chart hinges on, so it goes above it as
    # body text rather than into the caption. st.warning takes no HTML and the
    # hovers are worth more than the yellow box.
    _fn_line(
        f"\u26a0\ufe0f **Tracked data centers only** \u2014 {us_n} of "
        f"{us_n + rest} catalogued sites are American, and coverage elsewhere "
        "is thinner than the buildout. Non-US shares are floors.",
        ("coverage elsewhere is thinner than the buildout",
         "Sites outside the US are less often announced, permitted in public "
         "or sized in the press. A region that looks small here may just be "
         "one Epoch has not catalogued \u2014 missing coverage, not missing "
         "concrete."),
        ("Non-US shares are floors",
         "So the US share is a ceiling. Read a non-US line's movement over "
         "time rather than its level."))

    fig = go.Figure()
    if include_future and cap_date > today:
        # The shading sits under the filled areas; the divider and its labels
        # are what read on this chart.
        _dc_add_projection_band(fig, today, grid[-1])
    for lab, pct in shares.items():
        color = _DC_REGION_COLORS.get(lab, "#999999")
        fig.add_trace(go.Scatter(
            x=grid, y=pct, name=lab, mode='lines', stackgroup='one',
            line=dict(width=0.5, color=color),
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},"
                      f"{int(color[5:7], 16)},0.75)",
            customdata=[_dc_fmt_value(v, kind) for v in levels[lab]],
            hovertemplate="%{y:.1f}% — %{customdata}<extra>" + lab + "</extra>"))
    # The axis ends on the last sampled month, not on cap_date: a strip of
    # empty axis past the last stacked point reads as capacity dropping out.
    fig.update_layout(**_dc_layout(False, "Share of catalogued capacity",
                                   grid[0], grid[-1], y_range=[0, 100],
                                   show_legend=True))
    fig.update_layout(hovermode='x unified')
    fig.update_yaxes(ticksuffix='%')
    st.plotly_chart(fig, use_container_width=True)

    _fn_caption(
        f"Share of every site's {_dc_share_label(metric_label, kind)}, "
        "summed by where the building stands. SEA, Europe/UK and Other are "
        "geographic buckets"
        + (", and past today it is planned buildout." if include_future
           and cap_date > today else "."),
        ("where the building stands",
         "Geography only \u2014 no host filter, no tenant attribution. DayOne "
         "Johor counts under SEA here, though the panel above also reads it "
         "as China-accessible."),
        ("SEA, Europe/UK and Other",
         "SEA is Malaysia, Indonesia, Singapore, Thailand, Vietnam and the "
         "Philippines; Europe/UK is the EU, the UK and the rest of Europe; "
         "everything left over is Other."))


def render_data_centers():
    _today = datetime.now()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Data Centers")
        metric_label = st.selectbox("Capacity metric", list(_DC_METRICS),
                                    key="dc_metric")
        cfg = _DC_METRICS[metric_label]
        log_scale = st.checkbox("Log scale", value=cfg["log"], key="dc_log")
        include_future = st.checkbox("Include planned future buildout",
                                     value=True, key="dc_future")
        # Each site can be dated to any of three milestones; the choice shifts
        # every point forward by that lead time.
        # A bookmarked URL can carry a timing label from an older build; drop
        # it rather than letting the selectbox raise on an unknown value.
        if st.session_state.get("dc_timing") not in _DC_TIMING_OPTIONS:
            st.session_state.pop("dc_timing", None)
        timing_label = st.selectbox(
            "Date points at", list(_DC_TIMING_OPTIONS), key="dc_timing")
        # A bookmarked label from an older build must not raise.
        if st.session_state.get("dc_pool_n") not in _DC_NETWORK_OPTIONS:
            st.session_state.pop("dc_pool_n", None)
        net_label = st.selectbox("Data centers networked together",
                                 list(_DC_NETWORK_OPTIONS), key="dc_pool_n")
        if st.session_state.get("dc_party") not in _DC_PARTY_OPTIONS:
            st.session_state.pop("dc_party", None)
        party_label = st.radio(
            "Attribute each site to", list(_DC_PARTY_OPTIONS), key="dc_party",
            help="Tenant credits a site to every user Epoch lists (Colossus 2 "
                 "counts for Anthropic, Cursor and SpaceXAI alike), falling "
                 "back to the owner; operator credits the owner alone.")
        with st.expander("Country projection"):
            cty_cones = st.checkbox("Show projection cones", key="dc_cty_cones",
                                    value=_DC_DEFAULTS["dc_cty_cones"])
            pace_label = st.radio("Extrapolate along", list(_DC_CTY_PACE_OPTIONS),
                                  key="dc_cty_pace")
            cty_since = st.radio(
                "Fit trend since", _DC_CTY_SINCE_YEARS, horizontal=True,
                index=_DC_CTY_SINCE_YEARS.index(_DC_DEFAULTS["dc_cty_since"]),
                key="dc_cty_since",
                help="Early windows run hot (ramp from zero); late ones run "
                     "cool (under-catalogued).")
        with st.expander("Projection range"):
            dc_start_year = st.radio(
                "Chart starts", _DC_START_YEARS, horizontal=True,
                index=_DC_START_YEARS.index(_DC_DEFAULTS["dc_start_year"]),
                key="dc_start_year")
            dc_end_year = st.radio(
                "Project through", _DC_END_YEARS, horizontal=True,
                index=_DC_END_YEARS.index(_DC_DEFAULTS["dc_end_year"]),
                key="dc_end_year", disabled=not include_future,
                help="Planned buildout dated past this year is dropped, and "
                     "the by-country extrapolation runs to its end.")
        if st.button("Reset", key="dc_reset"):
            for k in _DC_RESET_KEYS:
                st.session_state.pop(k, None)
            # Re-seed defaults so they win over URL re-hydration on the rerun.
            st.session_state.update(_DC_DEFAULTS)
            st.rerun()

    key = cfg["key"]
    kind = cfg["kind"]
    dc_view = _dc_with_party(dc_all, _DC_PARTY_OPTIONS[party_label])
    # Cap projected buildout at the end of the chosen projection year.
    cap_date = datetime(dc_end_year, 12, 31) if include_future else _today
    series = _dc_series_for_metric(dc_view, key, cap_date=cap_date)
    # Drop the colocation / neutral-host providers too small to matter; the big
    # ones are charted and marked instead (see _dc_hidden_companies).
    hidden = _dc_hidden_companies(dc_view, now=_today)
    unattributed = _dc_unattributed_companies(dc_view)
    series = {n: v for n, v in series.items() if v['company'] not in hidden}

    # Shift every data point forward from the site's availability date to the
    # chosen milestone (DC construction / training done / model release). DC
    # construction means no shift.
    shift_days = _dc_timing_shift(timing_label, cfg.get("run_days", _DAYS_2MO))
    # Hover milestones name a training run only for the train-OP metrics, and
    # then the run length that metric assumes (2mo or 6mo).
    hover_run_days = cfg.get("run_days") if key in ('train_flop', 'train_flop_6mo') else None
    if shift_days:
        shift = timedelta(days=shift_days)
        series = {n: {'company': v['company'],
                      'pts': [(d + shift, val) for d, val in v['pts']]}
                  for n, v in series.items()}
        cap_date = cap_date + shift

    # ── Header ──
    st.header("Frontier Data Centers Over Time")

    if key in ('train_flop', 'train_flop_6mo'):
        run_days = cfg.get("run_days", _DAYS_2MO)
        st.caption(
            f"Methodology: *{metric_label}* = log₁₀(peak 8-bit OP/s × "
            f"{run_days // 30}-month run × {_DC_UTILIZATION:.0%} utilization). "
            "Order-of-magnitude estimates.")
    elif key in ('gpt5s', 'mythos'):
        target, scale = (("2e25 FLOP", "GPT-5 scale") if key == 'gpt5s'
                         else ("1e27 FLOP", "Mythos scale"))
        st.caption(
            f"Methodology: time to train one {target} ({scale}) model = {target} "
            f"÷ (peak 8-bit OP/s × {_DC_UTILIZATION:.0%} utilization). Shorter = "
            "bigger site.")

    if not series:
        st.warning("No data available for this metric.")
        return

    # Who is on the charts, and how sure we are whose hardware it is. Built from
    # the companies actually plotted, so it stays true as the data moves.
    _peak_h100 = _dc_company_peak_h100(dc_view)
    _shown_hosts = sorted(
        {v['company'] for v in series.values()} & _DC_EXCLUDE_COMPANIES,
        key=lambda c: -_peak_h100.get(c, 0.0))
    _shown_unattr = [c for c in _shown_hosts if c in unattributed]
    if _shown_hosts:
        _fn_caption(
            "Scope: AI labs, plus colocation hosts with a "
            f"{_dc_fmt_value(_DC_EXCLUDE_MIN_H100, 'h100')}-H100e site within "
            "a year "
            f"({_and_list([_dc_co_label(c, unattributed) for c in _shown_hosts])})"
            + (" \u2014 \u2020 marks one with no recorded tenant"
               if _shown_unattr else "")
            + ". The Compute/capabilities/diffusion tab excludes all hosts.",
            *([("no recorded tenant", "The landlord is named, not a lab.")]
              if _shown_unattr else []),
            ("excludes all hosts", "Its frontier has to be attributable to a "
                                   "lab that ships models."))

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Largest single data center over time
    # ══════════════════════════════════════════════════════════════════════
    env = _dc_envelope(series)
    st.subheader("Largest single data center")
    if kind == 'traintime':
        # The axis is plotted in runs-per-2mo (so "largest" stays a max) but
        # labelled in training time, which runs the other way.
        st.caption("Axis note: the y-axis is ordered biggest-site-first, so the "
                   "labelled training time gets **shorter** as the line rises.")

    if env:
        cd, cv, cn, cco = env[-1]

    env_dates = [e[0] for e in env]
    env_vals = [e[1] for e in env]
    end_x = cap_date if cap_date is not None else (
        env_dates[-1] if env_dates else _today)
    # Focus the view on the AI buildout era; earlier milestones (land clearing
    # back to 2018) are off-screen but still carried forward into the window.
    x_start = datetime(dc_start_year, 1, 1)
    x_end = end_x + timedelta(days=30)

    fig1 = go.Figure()
    if include_future:
        _dc_add_projection_band(fig1, _today, x_end)
    # The frontier envelope as a step line: solid for actual, dashed past today.
    (a_x, a_y), (p_x, p_y) = _dc_split_at(env, _today, end_x)
    fig1.add_trace(go.Scatter(
        x=a_x, y=a_y, mode='lines',
        line=dict(color='#1F77B4', width=3, shape='hv'),
        hoverinfo='skip', showlegend=False,
    ))
    if p_x is not None:
        fig1.add_trace(go.Scatter(
            x=p_x, y=p_y, mode='lines',
            line=dict(color='#1F77B4', width=3, shape='hv', dash='dash'),
            hoverinfo='skip', showlegend=False,
        ))
    # Mark frontier events: labelled dots where the leading site *changes* (filled
    # for actual, hollow for projected), plus smaller hollow dots where the same
    # leader *scales up* to a new capacity. Both carry hovers.
    new_dots, scale_dots = [], []
    prev_name, prev_val = None, None
    for d, v, name, co in env:
        if name != prev_name:
            new_dots.append((d, v, name, co))
        elif prev_val is not None and v != prev_val:
            scale_dots.append((d, v, name, co))
        prev_name, prev_val = name, v

    def _marker_trace(pts, projected):
        if not pts:
            return
        fig1.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode='markers+text',
            marker=dict(color='white' if projected else '#1F77B4', size=10,
                        line=dict(color='#1F77B4', width=1.5)),
            text=[p[2] for p in pts], textposition='top center',
            textfont=dict(size=9, color='#1F77B4'),
            hovertext=[f"{p[2]} ({_dc_co_label(p[3], unattributed)})"
                       f"{' — planned' if projected else ''}<br>"
                       f"{_dc_fmt_value(p[1], kind)}<br>{_dc_milestone_dates(p[0], shift_days, hover_run_days)}"
                       for p in pts],
            hoverinfo='text', showlegend=False,
        ))

    def _scale_trace(pts):
        if not pts:
            return
        fig1.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode='markers',
            marker=dict(color='white', size=7,
                        line=dict(color='#1F77B4', width=1.5)),
            hovertext=[f"{p[2]} ({_dc_co_label(p[3], unattributed)})"
                       f" — scale-up"
                       f"{' — planned' if p[0] > _today else ''}<br>"
                       f"{_dc_fmt_value(p[1], kind)}<br>"
                       f"{_dc_milestone_dates(p[0], shift_days, hover_run_days)}"
                       for p in pts],
            hoverinfo='text', showlegend=False,
        ))

    _marker_trace([p for p in new_dots if p[0] <= _today], projected=False)
    _marker_trace([p for p in new_dots if p[0] > _today], projected=True)
    _scale_trace(scale_dots)
    if env:
        _peak_projected = include_future and cd > _today
        fig1.add_trace(go.Scatter(
            x=[end_x], y=[env_vals[-1]], mode='markers',
            marker=dict(color='white' if _peak_projected else '#1F77B4', size=13,
                        symbol='diamond', line=dict(color='#1F77B4', width=1.5)),
            hovertext=[f"{'Planned peak' if _peak_projected else 'Current'}: "
                       f"{cn} ({_dc_co_label(cco, unattributed)})<br>"
                       f"{_dc_fmt_value(cv, kind)}<br>"
                       f"{_dc_milestone_dates(cd, shift_days, hover_run_days)}"],
            hoverinfo='text', showlegend=False,
        ))
    fig1.update_layout(**_dc_layout(
        log_scale, metric_label, x_start, x_end,
        y_range=_dc_yrange(_dc_visible_vals(env, x_start), log_scale),
        kind=kind))
    st.plotly_chart(fig1, use_container_width=True)

    comp = _dc_company_series(series)
    peaks = {co: max(v for _, v, _ in steps)
             for co, steps in comp.items() if steps}
    ranked = sorted(peaks, key=lambda c: peaks[c], reverse=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Current largest single data center by company (snapshot bar chart)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Current largest single data center by company")
    st.caption("Each company's single biggest site as of today — not the sum of "
               "all its sites, and excluding any planned / under-construction "
               "buildout.")

    # Snapshot strictly as of today: the latest step dated on or before _today,
    # so planned future buildout never inflates the current bars.
    def _step_at_today(steps):
        cur = None
        for s in steps:
            if s[0] <= _today:
                cur = s
            else:
                break
        return cur

    snap = []
    for co in comp:
        s = _step_at_today(comp[co])
        if s is not None:
            snap.append((co, s[1], s[2]))
    snap.sort(key=lambda t: t[1], reverse=True)
    fig_snap = go.Figure()
    fig_snap.add_trace(go.Bar(
        x=[s[1] for s in snap],
        y=[_dc_co_label(s[0], unattributed) for s in snap],
        orientation='h',
        marker=dict(color=[_dc_color(s[0], i) for i, s in enumerate(snap)]),
        text=[_dc_fmt_value(s[1], kind) for s in snap],
        textposition='outside',
        hovertext=[f"{_dc_co_label(s[0], unattributed)} — {s[2]}<br>"
                   f"{_dc_fmt_value(s[1], kind)}" for s in snap],
        hoverinfo='text',
    ))
    snap_xaxis = dict(title_text=f"Current {metric_label}",
                      type='log' if log_scale else 'linear',
                      gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222222'),
                      title_font=dict(color='#222222'))
    snap_range = _dc_yrange([s[1] for s in snap], log_scale)
    ticks = _dc_axis_ticks(snap_range, log_scale, kind)
    if ticks is not None:
        snap_xaxis.update(tickmode='array', tickvals=ticks[0],
                          ticktext=ticks[1])
    fig_snap.update_layout(
        height=max(300, 38 * len(snap) + 80),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=120, r=70, t=10, b=40),
        font=dict(color='#222222'), showlegend=False,
        xaxis=snap_xaxis,
        yaxis=dict(autorange='reversed', tickfont=dict(color='#222222')),
    )
    st.plotly_chart(fig_snap, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Largest single data center by company over time
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Largest single data center by company over time")
    st.caption("Each line is a company's biggest single site at that time "
               "(not the sum of all its sites). Solid = actual; dashed = planned "
               "/ under construction.")

    fig2 = go.Figure()
    if include_future:
        _dc_add_projection_band(fig2, _today, x_end)
    for i, co in enumerate(ranked):
        color = _dc_color(co, i)
        # The legend shows the marked label; grouping stays on the raw name.
        co_label = _dc_co_label(co, unattributed)
        steps = comp[co]
        (a_x, a_y), (p_x, p_y) = _dc_split_at(steps, _today, end_x)
        # Solid actual / dashed projected lines (hover handled by the dots).
        fig2.add_trace(go.Scatter(
            x=a_x, y=a_y, mode='lines',
            line=dict(color=color, width=2.5, shape='hv'),
            name=co_label, legendgroup=co, hoverinfo='skip',
        ))
        if p_x is not None:
            fig2.add_trace(go.Scatter(
                x=p_x, y=p_y, mode='lines',
                line=dict(color=color, width=2.5, shape='hv', dash='dash'),
                name=co_label, legendgroup=co, showlegend=False,
                hoverinfo='skip',
            ))
        # Filled dots where a *new* data center becomes the company's largest;
        # hollow dots where the same leading site scales up to a new capacity.
        new_dots, scale_dots = [], []
        prev_name, prev_val = None, None
        for s in steps:
            if s[2] != prev_name:
                new_dots.append(s)
            elif prev_val is not None and s[1] != prev_val:
                scale_dots.append(s)
            prev_name, prev_val = s[2], s[1]

        def _dot_trace(pts, hollow, label):
            if not pts:
                return
            fig2.add_trace(go.Scatter(
                x=[s[0] for s in pts], y=[s[1] for s in pts],
                mode='markers',
                marker=dict(size=6,
                            color='white' if hollow else color,
                            line=dict(color=color, width=1.5)),
                name=co_label, legendgroup=co, showlegend=False,
                hovertext=[
                    f"{co_label} — {s[2]} ({label})"
                    f"{' (planned)' if s[0] > _today else ''}<br>"
                    f"{_dc_fmt_value(s[1], kind)}<br>{_dc_milestone_dates(s[0], shift_days, hover_run_days)}"
                    for s in pts],
                hoverinfo='text',
            ))

        _dot_trace(new_dots, hollow=False, label="new leading site")
        _dot_trace(scale_dots, hollow=True, label="scale-up")
    comp_vals = [v for steps in comp.values()
                 for v in _dc_visible_vals(steps, x_start)]
    fig2.update_layout(**_dc_layout(log_scale, metric_label, x_start, x_end,
                                    y_range=_dc_yrange(comp_vals, log_scale),
                                    height=500, show_legend=True, kind=kind))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Data: Epoch AI, ‘Frontier Data Centers’ "
               "(epoch.ai/data/data-centers), CC-BY 4.0.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 5: Largest company capacity when several sites are networked
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Largest data center by company over time "
                 "(including networking multiple data centers)")
    st.caption(
        "Each line is the **sum** of a company's biggest group of sites that "
        "could be networked into one training job; the selector sets what "
        "counts as a group. Solid = actual; dashed = planned.")

    basis = _DC_NETWORK_OPTIONS[net_label]
    if basis == 'all':
        cluster_of = None
    elif basis == 'none':
        cluster_of = {}
    else:
        cluster_of = _dc_network_site_clusters(level=basis)
    pooled = _dc_company_networked_series(series, cluster_of)
    pooled = {co: steps for co, steps in pooled.items() if steps}
    pool_ranked = sorted(pooled,
                         key=lambda c: max(v for _, v, _, _ in pooled[c]),
                         reverse=True)

    fig_pool = go.Figure()
    if include_future:
        _dc_add_projection_band(fig_pool, _today, x_end)
    for i, co in enumerate(pool_ranked):
        color = _dc_color(co, i)
        co_label = _dc_co_label(co, unattributed)
        steps = pooled[co]
        (a_x, a_y), (p_x, p_y) = _dc_split_at(steps, _today, end_x)
        fig_pool.add_trace(go.Scatter(
            x=a_x, y=a_y, mode='lines',
            line=dict(color=color, width=2.5, shape='hv'),
            name=co_label, legendgroup=co, hoverinfo='skip',
        ))
        if p_x is not None:
            fig_pool.add_trace(go.Scatter(
                x=p_x, y=p_y, mode='lines',
                line=dict(color=color, width=2.5, shape='hv', dash='dash'),
                name=co_label, legendgroup=co, showlegend=False,
                hoverinfo='skip',
            ))
        # One dot per capacity change; the hover names the cluster and the sites
        # in it, so it is clear which buildings the line is adding together.
        dots = [s for j, s in enumerate(steps)
                if j == 0 or s[1] != steps[j - 1][1]]
        fig_pool.add_trace(go.Scatter(
            x=[s[0] for s in dots], y=[s[1] for s in dots], mode='markers',
            marker=dict(size=6, color=color, line=dict(color=color, width=1.5)),
            name=co_label, legendgroup=co, showlegend=False,
            hovertext=[
                f"{co_label}{' (planned)' if s[0] > _today else ''}<br>"
                f"{_dc_fmt_value(s[1], kind)}"
                f"{f' — {s[3]}, {len(s[2])} sites' if s[3] else ''}"
                "<br>" + "<br>".join(f"• {n}" for n in s[2]) + "<br>"
                f"{_dc_milestone_dates(s[0], shift_days, hover_run_days)}"
                for s in dots],
            hoverinfo='text',
        ))
    pool_vals = [v for steps in pooled.values()
                 for v in _dc_visible_vals(steps, x_start)]
    fig_pool.update_layout(**_dc_layout(log_scale, metric_label, x_start, x_end,
                                        y_range=_dc_yrange(pool_vals, log_scale),
                                        height=500, show_legend=True, kind=kind))
    st.plotly_chart(fig_pool, use_container_width=True)

    def _groups(basis):
        return ", ".join(lab for lab, b, _ in _DC_NETWORK_CLUSTERS if b == basis)

    if basis == 'none':
        scope = "No pooling: the chart above, redrawn. "
    elif basis == 'all':
        scope = "Every site a company has, summed — an upper bound, not a runnable job. "
    else:
        # List exactly the bases this level pools on, no more.
        scope = f"Curated groups. By proximity: {_groups('proximity')}. "
        if basis in ('fabric', 'plausible'):
            scope += f"By announced fabric: {_groups('fabric')}. "
        if basis == 'plausible':
            scope += (f"By plausible fabric (no link announced): "
                      f"{_groups('plausible')}. ")
        scope += "Everything else stands alone. "
    _fn_caption(
        scope.strip() + (" A shared site counts under each listed user."
                         if _DC_PARTY_OPTIONS[party_label] == 'tenant' else
                         " Every site is credited to the building's owner alone."),
        ("A shared site counts under each listed user",
         "So lines aren't additive across companies; unnamed tenants' halls fall "
         "to the landlord (\u2020).")
        if _DC_PARTY_OPTIONS[party_label] == 'tenant' else
        ("Every site is credited to the building's owner alone",
         "The tenant view instead counts a shared site under each listed user."))

    # By country: US vs China, extrapolated past the end of the recorded data.
    # Rebuilt without the host filter — country is about buildings, not tenants.
    _cty_series = _dc_series_for_metric(
        dc_view, key, cap_date=cap_date - timedelta(days=shift_days))
    if shift_days:
        _cty_series = {n: {'company': v['company'],
                           'companies': v.get('companies', [v['company']]),
                           'pts': [(d + timedelta(days=shift_days), val)
                                   for d, val in v['pts']]}
                       for n, v in _cty_series.items()}
    _dc_render_country_panel(
        _cty_series, {dc['name']: _dc_site_country(dc) for dc in dc_view},
        cluster_of, dcs=dc_view, today=_today, cap_date=cap_date, x_start=x_start,
        metric_label=metric_label, kind=kind, log_scale=log_scale,
        shift_days=shift_days, include_future=include_future,
        pace_mode=_DC_CTY_PACE_OPTIONS[pace_label], since=cty_since,
        horizon=dc_end_year, show_cones=cty_cones, run_days=hover_run_days)

    # Per-company: does the buildout predict releases?
    _cc_company_buildout(_today, cfg["key"], kind)

    # Where the capacity is: share of the catalogued total by region. Same
    # unfiltered, geography-only series the country panel reads.
    _dc_render_region_share(
        _cty_series, {dc['name']: _dc_site_country(dc) for dc in dc_view},
        today=_today, cap_date=cap_date, x_start=x_start,
        metric_label=metric_label, kind=kind, include_future=include_future)


# ══════════════════════════════════════════════════════════════════════════
# Compute/capabilities/diffusion — does data-center FLOP predict ECI?
# ══════════════════════════════════════════════════════════════════════════

def _cc_logop_yaxis(fig, title):
    """Label a hand-built compute figure's log y-axis in log₁₀ operations.

    The Data Centers tab routes its charts through `_dc_layout(kind='flop')`;
    these figures build their axis inline, so they get the same treatment here:
    the plotted value stays the raw operation count and only the tick text is
    logged. Tick positions are read off whatever the traces actually cover, and
    no explicit range is set, so plotly keeps autoscaling.
    """
    vals = []
    for tr in fig.data:
        for y in (getattr(tr, 'y', None) or []):
            if isinstance(y, (int, float)) and np.isfinite(y) and y > 0:
                vals.append(y)
    ticks = (_dc_logop_ticks([np.log10(min(vals)) - 0.05,
                              np.log10(max(vals)) + 0.05], True)
             if vals else None)
    ax = dict(title_text=title, type='log', gridcolor='rgba(0,0,0,0.12)',
              tickfont=dict(color='#222'), title_font=dict(color='#222'))
    if ticks is not None:
        ax.update(tickmode='array', tickvals=ticks[0], ticktext=ticks[1])
    fig.update_yaxes(**ax)


def _cc_trainflop_frontier(dcs, cap_date, with_names=False,
                           key='train_flop', run_days=_DAYS_2MO):
    """Running-max frontier of train-FLOP across AI-lab sites.

    `key`/`run_days` pick the training-run window — the loader's 2-month
    'train_flop' (default) or 6-month 'train_flop_6mo' column, exactly the
    DC/Pacing tabs' metric pair. Returns [(date, flop), …] sorted by date,
    with every point shifted forward by one training run (a run on a site
    available at D only finishes at D+run), matching the Data Centers tab.
    Monotonic non-decreasing.

    Every company in _DC_EXCLUDE_COMPANIES is dropped here **unconditionally**,
    which is deliberately stricter than the Data Centers tab: that tab now
    charts the big neutral hosts (_dc_hidden_companies), because a chart of
    where the compute is should show a 3.7M-H100 site whoever the landlord is.
    This frontier is not that. It is the compute half of a compute-vs-capability
    comparison, so every point has to be attributable to a lab that ships
    models — and QTS and DayOne sites have no recorded tenant at all. Crediting
    their capacity to nobody would raise the frontier without any capability to
    match it, distorting the fitted rates and China's ETA downstream. The two
    frontiers therefore differ on purpose; the Data Centers tab says so.
    """
    series = _dc_series_for_metric(dcs, key, cap_date=cap_date)
    series = {n: v for n, v in series.items()
              if v['company'] not in _DC_EXCLUDE_COMPANIES}
    # The same milestone the DC tab's "Date points at" selector calls
    # "Training run finished": capacity online at D trains a model by D+run.
    shift = timedelta(days=_dc_timing_shift("Training run finished", run_days))
    series = {n: {'company': v['company'],
                  'pts': [(d + shift, val) for d, val in v['pts']]}
              for n, v in series.items()}
    env = _dc_envelope(series)
    out = []
    best = 0.0
    best_name = None
    best_date = None       # date the current record was set (a fixed expansion)
    for d, v, name, _co in env:
        if v is None or v <= 0:
            continue
        if v > best:
            best, best_name, best_date = v, name, d
        out.append((d, best, best_name, best_date) if with_names else (d, best))
    return out


def _cc_loglinear_slope(pts):
    """OLS slope of log10(value) on years for [(date, value>0), …].

    Returns (slope_oom_per_year, intercept, year0_datetime) or None if <2 points.
    """
    usable = [(d, v) for d, v in pts if v is not None and v > 0]
    if len(usable) < 2:
        return None
    d0 = usable[0][0]
    x = np.array([(d - d0).days / 365.25 for d, _ in usable])
    y = np.array([np.log10(v) for _, v in usable])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), d0


def _cc_segment_fits(frontier_pts, today):
    """Per-segment growth of the FLOP frontier.

    Returns a list of dicts: {label, start, end, n, slope_oom, mult, doubling_mo,
    fit_x:[d0,d1], fit_y:[v0,v1]} — one per _CC_SEGMENTS entry with ≥2 points.
    """
    fits = []
    for label, start, end in _CC_SEGMENTS:
        seg_start = start if start is not None else today
        seg_end = end if end is not None else today
        seg = [(d, v) for d, v in frontier_pts if seg_start <= d <= seg_end]
        fit = _cc_loglinear_slope(seg)
        if fit is None:
            continue
        slope, intercept, d0 = fit
        mult = 10.0 ** slope
        doubling_mo = (12.0 * np.log10(2) / slope) if slope > 0 else float('inf')
        # Two endpoints of the fit line for plotting.
        xs = [seg[0][0], seg[-1][0]]
        ys = [10.0 ** (intercept + slope * ((d - d0).days / 365.25)) for d in xs]
        fits.append({
            'label': label, 'start': xs[0], 'end': xs[1], 'n': len(seg),
            'slope_oom': slope, 'mult': mult, 'doubling_mo': doubling_mo,
            'fit_x': xs, 'fit_y': ys,
        })
    return fits


def _cc_decomp(rows):
    """Regress ECI on log10(training FLOP) and time; decompose frontier growth.

    Returns a dict of fit statistics, or None if too few models. `a_partial` is
    the time-controlled compute coefficient (ECI per 10× FLOP), `a_solo` is the
    compute-only coefficient, `b_time` is ECI/year holding compute fixed.
    `frontier_compute_oom` and `eci_frontier_slope` describe the running-max-ECI
    frontier subset.
    """
    if len(rows) < 10:
        return None
    d0 = rows[0]['date']
    lc = np.array([m['log10_flop'] for m in rows])
    eci = np.array([m['eci'] for m in rows])
    t = np.array([(m['date'] - d0).days / 365.25 for m in rows])

    def _ols(X, y):
        X1 = np.column_stack([X, np.ones(len(y))])
        beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
        yh = X1 @ beta
        ss_res = float(((y - yh) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        return beta, r2

    beta_c, r2_c = _ols(lc, eci)                       # ECI ~ compute
    beta_t, r2_t = _ols(t, eci)                        # ECI ~ time
    beta_j, r2_j = _ols(np.column_stack([lc, t]), eci)  # ECI ~ compute + time

    # Frontier subset (running-max ECI) growth rates.
    fr = [m for m in rows if m.get('is_eci_frontier')]
    fr_compute_oom = None
    eci_frontier_slope = None
    if len(fr) >= 2:
        df0 = fr[0]['date']
        xf = np.array([(m['date'] - df0).days / 365.25 for m in fr])
        fr_compute_oom = float(np.polyfit(xf, np.array([m['log10_flop'] for m in fr]), 1)[0])
        eci_frontier_slope = float(np.polyfit(xf, np.array([m['eci'] for m in fr]), 1)[0])

    return {
        'n': len(rows),
        'a_solo': float(beta_c[0]), 'r2_compute': r2_c,
        'b_time_solo': float(beta_t[0]), 'r2_time': r2_t,
        'a_partial': float(beta_j[0]), 'b_time': float(beta_j[1]), 'r2_joint': r2_j,
        'corr_compute_time': float(np.corrcoef(lc, t)[0, 1]),
        'frontier_compute_oom': fr_compute_oom,
        'eci_frontier_slope': eci_frontier_slope,
        'n_frontier': len(fr),
        'flop_min': float(lc.min()), 'flop_max': float(lc.max()),
    }


# Iso-ECI capability bands used for the algorithmic-efficiency fits.
_CC_BANDS = [105, 115, 125]
_CC_BAND_HALFWIDTH = 4.0

# Iso-compute bands (log10 FLOP centers, ± half-dex) for the mirror-image view:
# hold the compute budget fixed and watch ECI climb over time.
_CC_COMPUTE_BANDS = [23.5, 24.5, 25.5]
_CC_CBAND_HALFWIDTH = 0.5


def _cc_efficiency(rows):
    """Algorithmic efficiency: how fast the compute for a *fixed* ECI falls.

    Fits the inverse regression log10(FLOP) = α·ECI + βₜ·t + c (compute as the
    outcome), so −βₜ is the order-of-magnitude/year reduction in compute needed
    to hold capability constant, and 1/α is ECI points per 10× compute. Also
    fits per-band iso-ECI lines for the chart. Returns None if too few models.
    """
    if len(rows) < 10:
        return None
    d0 = rows[0]['date']
    eci = np.array([m['eci'] for m in rows])
    t = np.array([(m['date'] - d0).days / 365.25 for m in rows])
    lc = np.array([m['log10_flop'] for m in rows])

    X = np.column_stack([eci, t, np.ones(len(rows))])
    beta, _, _, _ = np.linalg.lstsq(X, lc, rcond=None)
    yh = X @ beta
    ss_tot = float(((lc - lc.mean()) ** 2).sum())
    r2 = 1 - float(((lc - yh) ** 2).sum()) / ss_tot if ss_tot > 0 else float('nan')
    alpha = float(beta[0])
    g_inv = -float(beta[1])                 # OOM/yr, all-data inverse-regression

    bands = []
    band_slopes = []
    for center in _CC_BANDS:
        members = [m for m in rows if abs(m['eci'] - center) <= _CC_BAND_HALFWIDTH]
        if len(members) < 5:
            continue
        bx = np.array([(m['date'] - d0).days / 365.25 for m in members])
        by = np.array([m['log10_flop'] for m in members])
        s, b = np.polyfit(bx, by, 1)
        if s >= 0:                          # noisy edge band: skip the fit line
            continue
        xs = [members[0]['date'], members[-1]['date']]
        ys = [b + s * ((d - d0).days / 365.25) for d in xs]
        bands.append({'center': center, 'n': len(members), 'slope': float(s),
                      'fit_x': xs, 'fit_y': ys})
        band_slopes.append(-float(s))

    # Central estimate: average the all-data inverse fit with the band median,
    # which are the two non-dilution-inflated views. Report the spread too.
    band_med = float(np.median(band_slopes)) if band_slopes else g_inv
    g_central = (g_inv + band_med) / 2.0
    g_lo, g_hi = min(g_inv, band_med), max(g_inv, band_med)

    def _months(factor, g):
        return float(np.log10(factor) / g * 12) if g > 0 else float('inf')

    times = {f: {'central': _months(f, g_central),
                 'lo': _months(f, g_hi),     # more efficiency → fewer months
                 'hi': _months(f, g_lo)}
             for f in (2, 5, 10)}

    return {
        'n': len(rows), 'alpha': alpha, 'eci_per_oom': 1.0 / alpha if alpha else float('nan'),
        'g_inv': g_inv, 'band_median': band_med, 'g_central': g_central,
        'g_lo': g_lo, 'g_hi': g_hi, 'algo_mult': 10.0 ** g_central, 'r2': r2,
        'bands': bands, 'times': times,
    }


def _cc_iso_compute(rows):
    """Mirror of _cc_efficiency: hold compute fixed, watch ECI rise over time.

    Within each compute band (log10 FLOP ± half-dex) fits ECI = s·t + b, so s is
    ECI points/year of capability gain at a *constant* compute budget. Returns
    per-band fit lines plus the median rate, or None if too few models.
    """
    if len(rows) < 10:
        return None
    d0 = rows[0]['date']
    bands = []
    slopes = []
    for clog in _CC_COMPUTE_BANDS:
        mem = [m for m in rows if abs(m['log10_flop'] - clog) <= _CC_CBAND_HALFWIDTH]
        if len(mem) < 5:
            continue
        bx = np.array([(m['date'] - d0).days / 365.25 for m in mem])
        by = np.array([m['eci'] for m in mem])
        s, b = np.polyfit(bx, by, 1)
        if s <= 0:
            continue
        xs = [mem[0]['date'], mem[-1]['date']]
        ys = [b + s * ((d - d0).days / 365.25) for d in xs]
        bands.append({'center': clog, 'n': len(mem), 'slope': float(s),
                      'fit_x': xs, 'fit_y': ys})
        slopes.append(float(s))
    if not bands:
        return None
    return {'bands': bands, 'eci_per_yr': float(np.median(slopes)),
            'lo': float(min(slopes)), 'hi': float(max(slopes))}


def _cc_iso_compute_rate(rows, country, halfwidth=0.4):
    """Empirical algorithmic rate for one country: ECI/yr at a *fixed* compute
    budget.

    China's training compute is nearly flat (it clusters tightly at ~10^24.6
    FLOP), so a joint ECI-on-compute+time OLS can't separate the compute and
    algorithm terms — they're collinear. Holding compute roughly constant by
    construction (a band around the country's median log10 FLOP) sidesteps that:
    the resulting ECI-vs-time slope is a clean read of capability gained per year
    at a constant compute budget — i.e. the algorithmic-efficiency rate. Returns
    (rate ECI/yr, n_models, median_log10_flop) or (None, 0, None) if too sparse.
    """
    sub = [m for m in rows if m.get('country') == country]
    if len(sub) < 8:
        return None, 0, None
    med = float(np.median([m['log10_flop'] for m in sub]))
    band = [m for m in sub if abs(m['log10_flop'] - med) <= halfwidth]
    if len(band) < 8:
        return None, 0, None
    d0 = min(m['date'] for m in band)
    x = np.array([(m['date'] - d0).days / 365.25 for m in band])
    y = np.array([m['eci'] for m in band])
    s, _ = np.polyfit(x, y, 1)
    return float(s), len(band), med


# Epoch's estimate of pure pretraining algorithmic efficiency (~doubling every
# 8-9 months) — the floor for frontier progress with no stronger teacher.
_CC_PRETRAIN_ALGO_OOM = 0.4
# How long the already-published methods stock keeps paying out after a pause
# stops new publications — the diffusion channel's absorption ramp.
_CC_DIFF_ABSORB_YRS = 1.0


# Frontier-grade compute screen (OOM below the running-max training run).
# Near-frontier ECI reached well below frontier compute is the distillation
# fingerprint itself — DeepSeek/Qwen/Kimi ride within ~5 ECI of the frontier
# at 10–100× less compute — so a capability margin alone lets the heaviest
# distillers into the "can't-distill" control.
_CC_FG_FLOP_MARGIN = 1.0


def _cc_frontier_grade_algo(cc_rows, eci_all, margin=5.0,
                            flop_margin=_CC_FG_FLOP_MARGIN):
    """The _cc_decomp regression on frontier-grade models only.

    Frontier-grade = within `margin` ECI of the running frontier at release
    AND trained within `flop_margin` OOM of the running-max training run —
    the subset least able to have leaned on a stronger teacher. `eci_all`
    must be the full-window series (`load_eci_frontier(full_window=True)`):
    the ECI tab's Feb-2024 cutoff would leave every earlier model with no
    frontier to compare against, and such rows are dropped here, never
    admitted (they used to be, which propped up a spurious two-way
    coefficient gradient). Returns {'a_partial', 'b_time', 'n'} or None.

    The distillation fingerprint that survives the fix is one-way: this
    refit's b_time runs ~2–3 ECI/yr below the all-model fit's (followers
    ride a teacher; the frontier cannot). a_partial does NOT rise —
    reasoning-era models reach the frontier at sub-frontier compute, so the
    frontier-grade compute slope is flatter than pooled, not steeper.
    """
    fr = sorted(eci_all, key=lambda m: m['date'])
    run_d, run_v, best = [], [], 0.0
    for m in fr:
        best = max(best, m['eci_score'])
        run_d.append(m['date'])
        run_v.append(best)
    if not run_d:
        return None

    def _fr_at(d):
        i = bisect.bisect_right(run_d, d) - 1
        return run_v[i] if i >= 0 else None

    rows_s = sorted(cc_rows, key=lambda m: m['date'])
    fd, fv, mx = [], [], -float('inf')
    for m in rows_s:
        mx = max(mx, m['log10_flop'])
        fd.append(m['date'])
        fv.append(mx)

    def _flopmax_at(d):
        return fv[bisect.bisect_right(fd, d) - 1]

    sub = []
    for r in rows_s:
        f = _fr_at(r['date'])
        if f is None:
            continue                    # predates the frontier series
        if (r['eci'] >= f - margin
                and r['log10_flop'] >= _flopmax_at(r['date']) - flop_margin):
            sub.append(r)
    dec = _cc_decomp(sub)
    if dec is None:
        return None
    return {'a_partial': float(dec['a_partial']),
            'b_time': float(dec['b_time']), 'n': len(sub)}


def _cc_cn_level_offset(cc_rows, halfwidth=0.5, min_each=6):
    """China's ECI premium at matched compute and date, in points.

    The country dummy in ECI ~ t + is_CN + log10(FLOP) over US+China models
    inside a band around China's median training compute, fit from the date
    both countries are present in the band. Competitive distillation shows
    up here as a *level* — points banked at a given compute budget — not as
    a faster iso-compute rate: the two countries' band slopes are
    statistically indistinguishable, so this offset is the measured size of
    the distillation edge. Returns (offset_pts, n) or None when either
    country is too thin in the band.
    """
    cn = [m for m in cc_rows if m.get('country') == 'China']
    us = [m for m in cc_rows if m.get('country') == 'United States of America']
    if len(cn) < min_each or len(us) < min_each:
        return None
    med = float(np.median([m['log10_flop'] for m in cn]))
    band = [m for m in cn + us if abs(m['log10_flop'] - med) <= halfwidth]
    b_cn = [m for m in band if m['country'] == 'China']
    b_us = [m for m in band if m['country'] != 'China']
    if len(b_cn) < min_each or len(b_us) < min_each:
        return None
    start = max(min(m['date'] for m in b_cn), min(m['date'] for m in b_us))
    band = [m for m in band if m['date'] >= start]
    b_cn = [m for m in band if m['country'] == 'China']
    b_us = [m for m in band if m['country'] != 'China']
    if len(b_cn) < min_each or len(b_us) < min_each:
        return None
    d0 = min(m['date'] for m in band)
    t = np.array([(m['date'] - d0).days / 365.25 for m in band])
    y = np.array([m['eci'] for m in band])
    c = np.array([1.0 if m['country'] == 'China' else 0.0 for m in band])
    lc = np.array([m['log10_flop'] for m in band])
    X = np.column_stack([t, c, lc, np.ones(len(band))])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[1]), len(band)


def _cc_cn_pace_band(cn_fr, r_central):
    """Pace-factor (lo, hi, obs) reality-checking a bottom-up rate against
    China's observed frontier slope over _CC_GAP_WINDOWS; obs is None when
    the observed slopes are unavailable. Shared by the CC China-crossing
    section and the Pacing US-pause panel so the two can't disagree."""
    obs = [s for s in (_cc_frontier_eci_slope(cn_fr, cut)
                       for _, cut in _CC_GAP_WINDOWS) if s and s > 0]
    if obs and r_central > 0:
        return (min(0.85, min(obs) / r_central),
                max(1.15, max(obs) / r_central), obs)
    return 0.85, 1.15, None


def _cc_innovation_algo_band(cc_rows, eci_all=None):
    """(lo, hi) ECI/yr for algorithmic progress with no stronger teacher.

    The measured iso-compute rates include distillation, which a true frontier
    cannot use. Two near-frontier measurements bracket the rate: lo = the
    tight (±3) frontier-grade refit's time coefficient — still
    teacher-adjacent, so even the low end is generous — floored at the
    pretraining-efficiency prior × the frontier-grade exchange rate; hi = the
    top iso-compute band's own rate (internal-teacher distillation remains,
    hence an upper bound). None when the fits are unavailable. `eci_all`,
    when given, must be the full-window frontier series — the default load.
    """
    isoc = _cc_iso_compute(cc_rows)
    dec = _cc_decomp(cc_rows)
    if isoc is None or dec is None or not isoc.get('bands'):
        return None
    top = max(isoc['bands'], key=lambda b: b['center'])
    hi = float(top['slope'])
    if eci_all is None:
        eci_all = load_eci_frontier(_mtime=_eci_mtime(), full_window=True)
    fg3 = _cc_frontier_grade_algo(cc_rows, eci_all, margin=3.0)
    a_ref = fg3['a_partial'] if fg3 else dec['a_partial']
    lo = _CC_PRETRAIN_ALGO_OOM * a_ref
    if fg3:
        lo = max(lo, fg3['b_time'])
    return (min(lo, hi), max(lo, hi))


def _cc_pure_innovation_band(cc_rows, eci_all=None):
    """(lo, hi) ECI/yr for innovation alone — no distillation *and* no
    diffusion of others' methods: the pretraining-efficiency prior × the
    frontier-grade exchange rate, up to the no-external-distillation band's
    own floor (which still includes diffusion, hence the ceiling). None when
    the underlying fits are unavailable."""
    band = _cc_innovation_algo_band(cc_rows, eci_all)
    dec = _cc_decomp(cc_rows)
    if band is None or dec is None:
        return None
    if eci_all is None:
        eci_all = load_eci_frontier(_mtime=_eci_mtime(), full_window=True)
    fg3 = _cc_frontier_grade_algo(cc_rows, eci_all, margin=3.0)
    a_ref = fg3['a_partial'] if fg3 else dec['a_partial']
    lo = _CC_PRETRAIN_ALGO_OOM * a_ref
    hi = band[0]
    return (min(lo, hi), max(lo, hi))


def _eci_to_metr_p50_min(eci_score, a=0):
    """ECI score → estimated METR p50 time-horizon in minutes (central, lo, hi).

    Same Anthropic-calibrated fit the Epoch ECI tab uses for its ECI→METR hover
    (`a=1` for Anthropic-style models). The frontier forecast isn't tied to one
    lab, so `a` defaults to 0 (org-neutral). lo/hi are the fit's ±band.
    """
    central = 2 ** (0.24 * eci_score + 0.76 * a - 28.68)
    return central, central * 0.66, central * 1.34


def _eci_to_metr_p80_min(eci_score, a=0):
    """ECI score → estimated METR p80 time-horizon in minutes (central, lo, hi).

    The p80 companion to `_eci_to_metr_p50_min`, using the Epoch ECI tab's p80
    fit. p80 is the more demanding reliability bar, so horizons are shorter.
    """
    central = 2 ** (0.23 * eci_score + 0.35 * a - 29.95)
    return central, central * 0.52, central * 1.48


def _cc_quarter_ends(start, end):
    """Quarter-end dates strictly after `start` and through `end` (inclusive)."""
    out = []
    for yr in range(start.year, end.year + 1):
        for mo, dy in ((3, 31), (6, 30), (9, 30), (12, 31)):
            qd = datetime(yr, mo, dy)
            if qd > start and qd <= end:
                out.append(qd)
    return out


def _cc_eci_forecast(cc_rows, frontier, today, obs_slope, g_recent, g_planned,
                     share_lo, share_mid, share_hi,
                     horizon=datetime(2029, 12, 31)):
    """Quarterly frontier-ECI projection to the sidebar's *Project through*
    year-end (default end-2029).

    Decomposes the frontier's ECI growth into a *physical-compute* component that
    rides the projected compute-frontier path (so it decelerates as the buildout
    matures) and a constant *algorithmic-efficiency* component, then Monte-Carlos
    over the contested physical/algo mix, compute-delivery, and pace to produce a
    fan chart and a quarter-by-quarter table.

    Model (per trajectory, anchored at today's running-max ECI):
        ECI(t) = ECI_now
                 + (share·obs_slope·pace / g_recent) · cmult · Δlog₁₀FLOP_proj(t)
                 + (1 − share)·obs_slope·pace · Δt
    where Δlog₁₀FLOP_proj(t) is read off the projected compute frontier (extended
    past its end at the planned-buildout rate). The compute coefficient is
    calibrated so that, at today's compute slope, the physical term reproduces the
    `share` fraction of the observed frontier ECI rate.
    """
    if obs_slope is None or obs_slope <= 0:
        st.info("Not enough frontier history to project ECI.")
        return
    g_recent = g_recent if (g_recent and g_recent > 0) else (g_planned or obs_slope)

    # Current frontier anchor: the true running-max ECI across *all* models, not
    # just the compute-having subset (the newest frontier models rarely disclose
    # training FLOP, so cc_rows would anchor on a stale model like GPT-5).
    eci_all = load_eci_frontier(_mtime=_eci_mtime())
    anchor = max(eci_all, key=lambda m: m['eci_score']) if eci_all \
        else max(cc_rows, key=lambda m: m['eci'])
    anchor_name = anchor.get('display_name') or anchor.get('name', '')
    eci_now = anchor.get('eci_score', anchor.get('eci'))

    # Projected compute path: interpolate the running-max FLOP frontier in log
    # space; past its right edge, extend at the planned-buildout OOM/yr.
    ford = np.array([d.toordinal() for d, _ in frontier], dtype=float)
    flog = np.array([np.log10(v) for _, v in frontier], dtype=float)

    def _logflop_at(dt_):
        o = float(dt_.toordinal())
        if o <= ford[0]:
            return float(flog[0])
        if o >= ford[-1]:
            return float(flog[-1] + g_planned * (o - ford[-1]) / 365.25)
        return float(np.interp(o, ford, flog))

    logflop_now = _logflop_at(today)

    qdates = _cc_quarter_ends(today, horizon)
    x_dates = [today] + qdates
    dt_yrs = np.array([(d - today).days / 365.25 for d in x_dates])
    dlog = np.array([_logflop_at(d) - logflop_now for d in x_dates])

    # Monte-Carlo over the three uncertainties the section already quantifies.
    s_lo, s_mid, s_hi = sorted([share_lo, share_mid, share_hi])
    if s_hi - s_lo < 1e-6:
        s_lo, s_hi = s_lo - 0.05, s_hi + 0.05
    s_mid = min(max(s_mid, s_lo), s_hi)

    N = N_SAMPLES
    share = np.random.triangular(s_lo, s_mid, s_hi, N)          # physical/algo mix
    pace = np.clip(np.random.normal(1.0, 0.12, N), 0.65, 1.35)  # overall pace
    cmult = np.random.triangular(0.5, 1.0, 1.15, N)             # compute delivery

    coef_phys = (share * obs_slope * pace / g_recent * cmult)[:, None]
    coef_algo = ((1.0 - share) * obs_slope * pace)[:, None]
    traj = eci_now + coef_phys * dlog[None, :] + coef_algo * dt_yrs[None, :]

    pct = {p: np.percentile(traj, p, axis=0) for p in (5, 10, 25, 50, 75, 90, 95)}

    base = '#6A3D9A'
    r, g, b = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
    fig = go.Figure()
    _dc_add_projection_band(fig, today, horizon)

    # Decompose the median line into its two engines. The physical-compute
    # contribution is the median compute coefficient × the projected Δlog FLOP
    # (so it curves with the buildout); the rest, back down to today's anchor, is
    # the algorithmic-efficiency contribution. Split so the two stacked regions
    # sum exactly to the median line.
    coef_phys_med = float(np.median(coef_phys))
    phys_contrib = coef_phys_med * dlog
    algo_top = list(pct[50] - phys_contrib)           # anchor + algorithmic only
    base_y = [eci_now] * len(x_dates)
    fig.add_trace(go.Scatter(
        x=x_dates + x_dates[::-1], y=algo_top + base_y[::-1],
        fill='toself', fillcolor='rgba(31,119,180,0.22)', line=dict(width=0),
        name='Algorithmic component', hoverinfo='skip', showlegend=True))
    fig.add_trace(go.Scatter(
        x=x_dates + x_dates[::-1], y=list(pct[50]) + algo_top[::-1],
        fill='toself', fillcolor='rgba(214,39,40,0.22)', line=dict(width=0),
        name='Compute component', hoverinfo='skip', showlegend=True))

    bands = [
        (pct[5], pct[95], f'rgba({r},{g},{b},0.08)', '90% CI'),
        (pct[10], pct[90], f'rgba({r},{g},{b},0.16)', '80% CI'),
        (pct[25], pct[75], f'rgba({r},{g},{b},0.26)', '50% CI'),
    ]
    for lo, hi, color, label in bands:
        fig.add_trace(go.Scatter(
            x=x_dates + x_dates[::-1], y=list(hi) + list(lo[::-1]),
            fill='toself', fillcolor=color, line=dict(width=0),
            name=label, hoverinfo='skip', showlegend=True))
    metr_cd = [[fmt_hrs(_eci_to_metr_p50_min(s)[0] / 60),
                fmt_hrs(_eci_to_metr_p80_min(s)[0] / 60)] for s in pct[50]]
    fig.add_trace(go.Scatter(
        x=x_dates, y=list(pct[50]), mode='lines',
        line=dict(color=base, width=2.5, dash='dash'),
        name='Median projection', customdata=metr_cd,
        hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}'
                      '<br>METR p50 ≈ %{customdata[0]}'
                      '<br>METR p80 ≈ %{customdata[1]}<extra></extra>'))
    # Historical frontier ECI points for context (true running-max frontier).
    fr = [m for m in eci_all if m.get('is_frontier')] or \
        [m for m in cc_rows if m.get('is_eci_frontier')]
    fr_get = (lambda m: (m.get('eci_score', m.get('eci')),
                         m.get('display_name') or m.get('name', '')))
    # Drop Claude 3 Sonnet (an early, low frontier point that stretches the axis).
    fr = [m for m in fr if 'claude 3 sonnet' not in fr_get(m)[1].lower()]
    fig.add_trace(go.Scatter(
        x=[m['date'] for m in fr], y=[fr_get(m)[0] for m in fr], mode='lines+markers',
        line=dict(color='#888', width=1.5),
        marker=dict(size=6, color=base, line=dict(color='white', width=0.5)),
        text=[f"{pretty(fr_get(m)[1])}<br>ECI {fr_get(m)[0]:.0f}" for m in fr],
        hoverinfo='text', name='ECI frontier (actual)'))
    fig.update_layout(
        height=460, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=55, r=20, t=20, b=40), font=dict(color='#222222'),
        legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)', range=[datetime(2024, 1, 1),
                   horizon], tickfont=dict(color='#222'),
                   title_font=dict(color='#222')),
        yaxis=dict(title_text="Frontier ECI score", gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222'), title_font=dict(color='#222')))
    st.plotly_chart(fig, use_container_width=True)
    phys_end = float(phys_contrib[-1])
    algo_end = float(pct[50][-1] - eci_now - phys_end)
    _fn_caption(
        f"The median rise splits into algorithms (~{algo_end:.0f} ECI by "
        f"end-{horizon.year}, the blue band) and physical compute "
        f"(~{phys_end:.0f}, the red band on top). Compute is the thinner, "
        "decelerating slice.",
        ("the blue band", "The algorithmic-efficiency contribution: a straight "
                          "climb at a constant per-year rate."),
        ("the red band", "The physical-compute contribution, which curves as it "
                         "rides the projected compute frontier and flattens as "
                         "the buildout matures."))

    # Year-end milestone cards, each with its estimated METR p50 horizon.
    _ms_years = list(range(today.year, horizon.year + 1))
    cols = st.columns(len(_ms_years))
    for col, yr in zip(cols, _ms_years):
        target = datetime(yr, 12, 31)
        j = min(range(len(x_dates)), key=lambda i: abs((x_dates[i] - target).days))
        col.metric(f"End {yr}", f"ECI {pct[50][j]:.0f}",
                   f"{pct[10][j]:.0f}–{pct[90][j]:.0f} (80%)")
        p50_c = _eci_to_metr_p50_min(pct[50][j])[0]
        p50_lo = _eci_to_metr_p50_min(pct[10][j])[0]
        p50_hi = _eci_to_metr_p50_min(pct[90][j])[0]
        p80_c = _eci_to_metr_p80_min(pct[50][j])[0]
        p80_lo = _eci_to_metr_p80_min(pct[10][j])[0]
        p80_hi = _eci_to_metr_p80_min(pct[90][j])[0]
        col.markdown(
            f"<div style='font-size:0.72em; color:#888; margin-top:-0.4em;'>"
            f"METR p50 ≈ {fmt_hrs(p50_c / 60)}<br>"
            f"({fmt_hrs(p50_lo / 60)}–{fmt_hrs(p50_hi / 60)}, 80%)<br>"
            f"METR p80 ≈ {fmt_hrs(p80_c / 60)}<br>"
            f"({fmt_hrs(p80_lo / 60)}–{fmt_hrs(p80_hi / 60)}, 80%)</div>",
            unsafe_allow_html=True)

    # Quarter-by-quarter table.
    with st.expander("Quarterly ECI projection table"):
        tmd = ["| Quarter | ECI | METR p50 | METR p80 |",
               "|---|---|---|---|"]
        for i, d in enumerate(x_dates):
            if i == 0:
                continue
            q = (d.month - 1) // 3 + 1
            p50_c = fmt_hrs(_eci_to_metr_p50_min(pct[50][i])[0] / 60)
            p50_lo = fmt_hrs(_eci_to_metr_p50_min(pct[10][i])[0] / 60)
            p50_hi = fmt_hrs(_eci_to_metr_p50_min(pct[90][i])[0] / 60)
            p80_c = fmt_hrs(_eci_to_metr_p80_min(pct[50][i])[0] / 60)
            p80_lo = fmt_hrs(_eci_to_metr_p80_min(pct[10][i])[0] / 60)
            p80_hi = fmt_hrs(_eci_to_metr_p80_min(pct[90][i])[0] / 60)
            tmd.append(
                f"| {d.year} Q{q} | "
                f"**{pct[50][i]:.0f}** ({pct[10][i]:.0f}–{pct[90][i]:.0f}) | "
                f"{p50_c} ({p50_lo}–{p50_hi}) | "
                f"{p80_c} ({p80_lo}–{p80_hi}) |")
        st.markdown("\n".join(tmd))
        st.caption("Each cell shows the median with its 80% band in parentheses.")

    _fn_caption(
        "Frontier ECI = an anchor plus compute growth plus a constant "
        "algorithmic term, sampled over its inputs, with METR horizons read off "
        "the ECI bridge. Order-of-magnitude, not a promise.",
        ("an anchor plus compute growth", "Today's anchor + physical-compute "
                                          f"growth (its share of ~{obs_slope:.0f} "
                                          "ECI/yr, scaled to the projected "
                                          "compute path so it decelerates with "
                                          "the buildout)."),
        ("sampled over its inputs", f"The physical/algo mix ({s_lo*100:.0f}"
                                    f"\u2013{s_hi*100:.0f}%), compute delivery "
                                    "(\u00d70.5\u20131.15 of plan) and pace "
                                    "(\u00b112%)."),
        ("the ECI bridge", "Org-neutral fits p50_min = 2^(0.24\u00b7ECI "
                           "\u2212 28.68) and p80_min = 2^(0.23\u00b7ECI "
                           "\u2212 29.95); their ranges track the ECI 80% band, "
                           "not the fits' own."),
        ("not a promise", "This extrapolates the historical linear ECI trend: no "
                          "paradigm shift, no compute wall, and ECI still "
                          "meaning something at these levels."))


def _cc_country_frontier(models, country):
    """Running-max ECI frontier for one country, as [(date, score, name), …].

    `models` is the load_eci_frontier() list (each has 'country', 'eci_score',
    'date', 'display_name'). We recompute the running max *within* the country so
    each side's own frontier is independent of the other's.
    """
    grp = sorted((m for m in models if m.get('country') == country),
                 key=lambda m: m['date'])
    out = []
    best = -float('inf')
    for m in grp:
        if m['eci_score'] > best:
            best = m['eci_score']
            out.append((m['date'], m['eci_score'],
                        m.get('display_name') or m.get('name', '')))
    return out


def _cc_country_compute_frontier(rows, country):
    """Running-max *training-compute* frontier for one country, from the models
    that disclose FLOP, as [(date, log10_flop, eci, model_name), …].

    This is the largest disclosed training run to date on each side — the compute
    trajectory itself, the thing export controls actually constrain. Also returns
    the OLS growth rate (OOM/yr) over that frontier.
    """
    grp = sorted((m for m in rows if m.get('country') == country),
                 key=lambda m: m['date'])
    pts = []
    best = -float('inf')
    for m in grp:
        if m['log10_flop'] > best:
            best = m['log10_flop']
            pts.append((m['date'], m['log10_flop'], m['eci'],
                        m.get('name') or 'frontier model'))
    if len(pts) < 2:
        return pts, None
    b = pts[0][0]
    g = float(np.polyfit(
        np.array([(d - b).days / 365.25 for d, lf, s, n in pts]),
        np.array([lf for d, lf, s, n in pts]), 1)[0])
    return pts, g


# ── Shared US-trend gap helper (also used by the ECI China tab) ───────────

def _eci_months_behind(us_fr, score, at_date):
    """Months `score` lags the US ECI frontier's linear trend, at `at_date`.

    `us_fr` is [(date, score, name), …] — a running-max US frontier. This is the
    same gap-in-time metric the ECI China tab reports, factored out so both
    places define it identically. Positive = behind the US trend; negative =
    ahead. Returns NaN if the US trend is flat/declining.
    """
    base = us_fr[0][0]
    days = np.array([(d - base).days for d, s, n in us_fr], dtype=float)
    scores = np.array([s for d, s, n in us_fr])
    intercept, slope = fit_line(days, scores)
    if slope <= 0:
        return float('nan')
    us_date = base + timedelta(days=float((score - intercept) / slope))
    return (at_date - us_date).days / 30.44


# Windows for the country frontier-slope sensitivity band. Central = the
# post-catch-up regime we commit to; the band spans the full record (China still
# closing the 2023 gap) to 2025-only (the US pulling ahead fastest).
_CC_GAP_WINDOWS = [
    ("full record", datetime(2023, 1, 1)),
    ("since 2024", datetime(2024, 1, 1)),
    ("2025 only", datetime(2025, 1, 1)),
]

# Forward compute-growth ranges per country, for the Chart-A cones.
#
# US is MEASURED: Epoch's data-center buildout (committed/under-construction) gives
# a data-backed range — the recent built pace (~3×/yr) down to the planned 2026–28
# pipeline (~2×/yr). Those segment slopes are read live from the DC data, so no
# US constant lives here.
#
# China is still a JUDGMENT call, but no longer unchecked. Epoch's catalogue now
# holds a handful of Chinese sites (VNET Ulanqab…) plus the China-accessible
# DayOne Johor campus, and the Data Centers tab's by-country panel projects
# them; _cc_country_pace_check() reads that same engine here. The band stays a
# hand-set constant rather than that fit because the two measure different
# things: the catalogue pace is *buildings* (and China-accessible is a from-zero
# ramp the DC tab itself flags as running hot), while this band is the growth of
# China's largest *coherent single-cluster training run* — capped by export
# controls on leading-edge chips and, more bindingly, on the networking to fuse
# dispersed chips into one run. Low end = controls bite / stockpiles deplete;
# high end = China sustains its most recent disclosed leg (~2×/yr). The band is
# kept at or below the catalogue's paces — TestCcCnComputeBand pins that against
# the live data; if a refresh breaks it, retarget these deliberately.
_CC_CN_COMPUTE_LO = 0.15   # ~1.4×/yr — export controls bite
_CC_CN_COMPUTE_HI = 0.30   # ~2×/yr — China's recent disclosed pace holds


def _cc_country_pace_check(today):
    """Largest-site train-FLOP pace per country, from the DC tab's own engine.

    Runs _dc_country_steps + _dc_cty_fit exactly as the Data Centers tab's
    by-country panel does (no pooling, fit window clipped at the plan horizon,
    unfiltered site list — a landlord's hall is capacity in its country), and
    returns {label: fit-or-None} for the US, China-accessible and
    China-domestic series. Used to cross-check this tab's segment fits (US)
    and the hand-set _CC_CN_COMPUTE band (China) in captions and tests, so
    the two tabs can't silently disagree about either country's compute pace.
    Always the 2-month 'train_flop' column: the 6-month values are a constant
    multiple, so the fitted paces this returns are identical either way.
    """
    series = _dc_series_for_metric(dc_all, 'train_flop')
    country_of = {dc['name']: _dc_site_country(dc) for dc in dc_all}
    groups = _dc_country_groups(series, country_of, 'abroad')
    groups[_DC_CTY_CN_DOMESTIC] = [n for n in series
                                   if country_of.get(n) == _DC_CTY_CN]
    t_end = today + timedelta(days=_DC_CTY_PLAN_HORIZON_DAYS)
    return {label: _dc_cty_fit(_dc_country_steps(series, groups.get(label, []),
                                                 'site', {}), t_end=t_end)
            for label in (_DC_CTY_US, _DC_CTY_CN_ACCESS, _DC_CTY_CN_DOMESTIC)}

# China cluster-capacity estimate — largest *single cluster's* 2-month training
# FLOP, to match the US buildout line (which is the largest single site, not a
# national total). No Epoch buildout data for China, so this is a judgment
# (mid-2026):
#  • China's *total* fleet is large — mostly smuggled / third-country-routed
#    NVIDIA (H100/H800/H20/B200) plus a domestic Huawei Ascend fleet (~250–300k
#    usable 910C in 2026, HBM-capped; SemiAnalysis). But total compute ≠ one
#    cluster: smuggled chips arrive dispersed, and the networking gear to fuse
#    them into one coherent training run is itself export-controlled.
#  • So the binding constraint is the largest *single* coherent cluster. We anchor
#    capacity on China's largest *demonstrated* run and add only modest headroom:
#    being compute-constrained, China likely uses most of its biggest cluster per
#    run (unlike the US, whose runs sit ~10× below capacity on efficiency slack).
_CC_CN_CAPACITY_HEADROOM_OOM = 0.5    # ~3× above the largest demonstrated run

# _CC_RUN_COMPLETION_LAG (model release lag) is defined up by the 2-month-run
# constants, since both the Data Centers tab and this section use it.


def _cc_frontier_eci_slope(fr, cutoff):
    """OLS ECI/yr of a country's running-max frontier from `cutoff` onward."""
    pts = [(d, s) for d, s, n in fr if d >= cutoff]
    if len(pts) < 2:
        return None
    base = pts[0][0]
    return float(np.polyfit(
        np.array([(d - base).days / 365.25 for d, _ in pts]),
        np.array([s for _, s in pts]), 1)[0])


def _cc_pooled_decomp(rows):
    """Pooled ECI decomposition from US+China models with disclosed FLOP.

    Joint OLS of ECI on log10(FLOP) and time, returning (a_partial, b_algo):
    a_partial = ECI per ×10 compute (the exchange rate), b_algo = ECI/yr at fixed
    compute (shared algorithmic progress, since methods/open weights diffuse).
    Compute and time are collinear, so the split is approximate. Returns
    (None, None) if too few models.
    """
    sub = [m for m in rows
           if m.get('country') in ('United States of America', 'China')]
    if len(sub) < 10:
        return None, None
    d0 = min(m['date'] for m in sub)
    lc = np.array([m['log10_flop'] for m in sub])
    t = np.array([(m['date'] - d0).days / 365.25 for m in sub])
    eci = np.array([m['eci'] for m in sub])
    beta, _, _, _ = np.linalg.lstsq(
        np.column_stack([lc, t, np.ones(len(sub))]), eci, rcond=None)
    return float(beta[0]), float(beta[1])


def _cc_us_vs_china(cc_rows, today, horizon=datetime(2029, 12, 31),
                    run_key='train_flop', run_days=_DAYS_2MO):
    """Section 4: the US-China frontier read through the compute lens.

    The honest headline is a *mismatch of scale*: the US holds a training-compute
    lead of orders of magnitude, but only a single-digit ECI lead. China closed
    most of the gap in one 2023 burst; since 2024 the US has edged back ahead
    slowly. Returns to compute are modest and algorithmic progress diffuses
    across the field, so a huge compute gap buys only a small, slowly-widening
    capability gap. We commit to the since-2024 regime and show the full-record /
    2025-only windows as the band, because the trend's sign depends on which you
    pick.
    """
    st.subheader("US vs. China")

    eci_all = load_eci_frontier(_mtime=_eci_mtime())
    us_fr = _cc_country_frontier(eci_all, 'United States of America')
    cn_fr = _cc_country_frontier(eci_all, 'China')
    us_cf, g_us = _cc_country_compute_frontier(cc_rows, 'United States of America')
    cn_cf, g_cn = _cc_country_compute_frontier(cc_rows, 'China')
    if len(us_fr) < 2 or len(cn_fr) < 2 or g_us is None or g_cn is None:
        st.info("Not enough country-tagged ECI/compute history to compare.")
        return

    us_best = max(us_fr, key=lambda x: x[1])
    cn_best = max(cn_fr, key=lambda x: x[1])
    gap_now = us_best[1] - cn_best[1]
    mo_now = _eci_months_behind(us_fr, cn_best[1], cn_best[0])

    # ── Compute: largest *actual* runs (grounded, Epoch per-model) vs cluster
    # *capacity* (US measured from buildout; China estimated from its chips). ──
    # Actual-run frontiers are us_cf / cn_cf (Epoch's per-model estimates). The
    # headline gap is run-to-run.
    us_run_lf, cn_run_lf = us_cf[-1][1], cn_cf[-1][1]
    run_gap_oom = us_run_lf - cn_run_lf

    # US capacity = Epoch's data-center buildout (the largest cluster's training
    # capacity over the sidebar's run window); its recent-built vs planned
    # segments give the forward range.
    dc_fr = _cc_trainflop_frontier(dc_all, horizon, with_names=True,
                                   key=run_key, run_days=run_days)
    dc_fits = _cc_segment_fits([(d, v) for d, v, n, sd in dc_fr], today)
    g_us_hi = next((f['slope_oom'] for f in dc_fits
                    if f['label'].startswith('2025 H2')), 0.47)   # recent built
    g_us_lo = next((f['slope_oom'] for f in dc_fits
                    if f['label'].startswith('Planned')), 0.27)   # planned pipeline
    # Keep every buildout point within the chart window — including the announced
    # megaclusters years out (Colossus 2, Hyperion, Stargate…) — so the labeled
    # line rides the real announced buildout to ~2029, not a flat extrapolation.
    us_cap_hist = [(d, np.log10(v), n, sd) for d, v, n, sd in dc_fr if d <= horizon]
    us_cap_lf = next(lf for d, lf, n, sd in reversed(us_cap_hist) if d <= today)
    us_cap_end_lf = us_cap_hist[-1][1]
    # China capacity = its largest *demonstrated* run + modest single-cluster
    # headroom, anchored at the last run so the fan connects to the data.
    # run_extra scales the band by the run-window ratio so both countries'
    # ceilings share the assumption (zero at the 2-month default).
    g_cn_lo, g_cn_hi = _CC_CN_COMPUTE_LO, _CC_CN_COMPUTE_HI
    run_extra = float(np.log10(run_days / _DAYS_2MO))
    cn_cap_d = cn_cf[-1][0] - _CC_RUN_COMPLETION_LAG
    cn_cap_apex_lo = cn_run_lf + run_extra
    cn_cap_apex_hi = cn_run_lf + _CC_CN_CAPACITY_HEADROOM_OOM + run_extra
    d_yrs = (today - cn_cap_d).days / 365.25
    cn_cap_lo_lf = cn_cap_apex_lo + g_cn_lo * d_yrs     # capacity range at today
    cn_cap_hi_lf = cn_cap_apex_hi + g_cn_hi * d_yrs
    cn_cap_lf = 0.5 * (cn_cap_lo_lf + cn_cap_hi_lf)
    cap_gap_oom = us_cap_lf - cn_cap_lf

    # ECI projection: derived from compute (Chart A growth) + shared algorithmic
    # progress. a_partial = ECI per ×10 compute; b_algo = shared ECI/yr at fixed
    # compute (methods diffuse). Each country's ECI slope = a_partial·g + b_algo.
    # Frontier projections use the frontier-grade coefficient pair — the
    # all-model fit's b_time is inflated by distillation among followers,
    # which the frontier cannot use, so the refit (near-frontier ECI at
    # frontier-scale compute, full-window frontier) runs ~2–3 ECI/yr slower
    # at fixed compute. The pooled fit is the fallback when the refit is too
    # thin.
    a_partial, b_algo = _cc_pooled_decomp(cc_rows)
    fgm = _cc_frontier_grade_algo(
        cc_rows, load_eci_frontier(_mtime=_eci_mtime(), full_window=True))
    if fgm:
        a_partial, b_algo = fgm['a_partial'], fgm['b_time']
    inno_band = _cc_innovation_algo_band(cc_rows)
    pure_band = _cc_pure_innovation_band(cc_rows)
    us_eci_slo, us_eci_shi = b_algo + a_partial * g_us_lo, b_algo + a_partial * g_us_hi
    cn_eci_slo, cn_eci_shi = b_algo + a_partial * g_cn_lo, b_algo + a_partial * g_cn_hi
    us_eci_smid = 0.5 * (us_eci_slo + us_eci_shi)
    cn_eci_smid = 0.5 * (cn_eci_slo + cn_eci_shi)

    dt_end = (horizon - today).days / 365.25
    gap_end = (us_best[1] + us_eci_smid * dt_end) - (cn_best[1] + cn_eci_smid * dt_end)
    mo_end = (gap_end / us_eci_smid * 12.0) if us_eci_smid > 0 else float('nan')

    c1, c2, c3 = st.columns(3)
    c1.metric("Compute gap (largest actual run)",
              f"~{10 ** run_gap_oom:.0f}×",
              f"{run_gap_oom:.1f} OOM, US ahead", delta_color="off")
    c2.metric("ECI gap today", f"{gap_now:.0f} pts",
              f"~{mo_now:.0f} mo behind", delta_color="off")
    c3.metric(f"ECI gap end-{horizon.year} (compute + algo)",
              f"~{gap_end:.0f} pts",
              f"~{mo_end:.0f} mo behind", delta_color="off")

    # ── Chart A: actual training runs (grounded) vs cluster capacity (est.) ────
    figc = go.Figure()
    _dc_add_projection_band(figc, today, horizon)

    # Project a capacity band forward: the [lo,hi] level uncertainty at the anchor
    # grows with the [g_lo,g_hi] rate uncertainty (so a wide band stays wide).
    def _cap_cone(anchor_d, lo_lf, hi_lf, g_lo, g_hi, rgb):
        yrs = (horizon - anchor_d).days / 365.25
        r, gg, bb = rgb
        figc.add_trace(go.Scatter(
            x=[anchor_d, horizon, horizon, anchor_d], mode='lines',
            y=[10.0 ** lo_lf, 10.0 ** (lo_lf + g_lo * yrs),
               10.0 ** (hi_lf + g_hi * yrs), 10.0 ** hi_lf],
            fill='toself', fillcolor=f'rgba({r},{gg},{bb},0.10)', line=dict(width=0),
            hoverinfo='skip', showlegend=False))

    # GROUNDED — largest actual estimated training runs (Epoch per-model), dated
    # at estimated training completion (release − ~1mo) so they line up with the
    # run-completion-dated capacity frontier.
    for cf, col, label in ((us_cf, '#1F77B4', 'US runs (Epoch est.)'),
                           (cn_cf, '#D62728', 'China runs (Epoch est.)')):
        figc.add_trace(go.Scatter(
            x=[d - _CC_RUN_COMPLETION_LAG for d, lf, s, n in cf],
            y=[10.0 ** lf for d, lf, s, n in cf],
            mode='lines+markers', line=dict(color=col, width=2.5),
            marker=dict(size=6, color=col, line=dict(color='white', width=0.5)),
            text=[f"<b>{n}</b><br>{_logop_lbl(lf)} &nbsp; ECI {s:.0f}<br>"
                  f"Released {d:%b %Y}<br><i>plotted at est. training completion "
                  f"(−{_CC_RUN_COMPLETION_LAG.days / 30.44:.1f}mo)</i>"
                  for d, lf, s, n in cf],
            hoverinfo='text', name=label))

    # ESTIMATED — cluster capacity ceilings (what the biggest cluster could
    # train), drawn faded/dashed so they read as estimates, not runs.
    # Hover dates come from `sd` — the date the leading site SET the record (a
    # fixed expansion) — not the running x-date, so they don't drift month to month.
    figc.add_trace(go.Scatter(
        x=[d for d, lf, n, sd in us_cap_hist],
        y=[10.0 ** lf for d, lf, n, sd in us_cap_hist],
        mode='lines', line=dict(color='#1F77B4', width=1.5, dash='dash'),
        opacity=0.7, name='US capacity (buildout)',
        text=[f"<b>{n}</b><br>{_logop_lbl(lf)} ({run_days // 30}-mo run)<br>"
              f"{_dc_milestone_dates(sd, run_days, run_days)}"
              for d, lf, n, sd in us_cap_hist],
        hoverinfo='text'))
    # US capacity fan emanates from the last US *run* (not the DC line): lower edge
    # at the run, upper edge +the run→announced-capacity gap, growing at the
    # buildout rate. The announced DC line falls inside it.
    us_run_d = us_cf[-1][0] - _CC_RUN_COMPLETION_LAG
    us_headroom = us_cap_lf - us_run_lf
    _cap_cone(us_run_d, us_run_lf, us_run_lf + us_headroom, g_us_lo, g_us_hi,
              (31, 119, 180))
    # China capacity fan: lower edge anchored at its largest demonstrated run (so
    # it connects to the last red dot), upper edge +headroom; fans by growth range.
    figc.add_trace(go.Scatter(
        x=[today, today], y=[10.0 ** cn_cap_lo_lf, 10.0 ** cn_cap_hi_lf],
        mode='lines', line=dict(color='#D62728', width=7), opacity=0.3,
        hoverinfo='skip', name='China capacity (est.)'))
    _cap_cone(cn_cap_d, cn_cap_apex_lo, cn_cap_apex_hi, g_cn_lo, g_cn_hi, (214, 39, 40))

    figc.update_layout(
        height=420, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=20, t=20, b=40), font=dict(color='#222222'),
        legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)',
                   range=[datetime(2023, 1, 1), horizon],
                   tickfont=dict(color='#222'), title_font=dict(color='#222')),
        yaxis=dict(gridcolor='rgba(0,0,0,0.12)'))
    _cc_logop_yaxis(figc, "Training compute (log₁₀ OP)")
    st.plotly_chart(figc, use_container_width=True)
    _fn_caption(
        f"<b>Solid</b> = largest actual runs Epoch estimates: "
        f"~{_logop_lbl(us_run_lf)} (US) vs ~{_logop_num(cn_run_lf)} (China), a "
        f"<b>~{10 ** run_gap_oom:.0f}\u00d7 ({run_gap_oom:.1f} OOM)</b> gap. "
        f"<b>Dashed/shaded</b> = capacity, the largest single cluster's "
        f"{run_days // 30}-month run \u2014 and the gap widens only slowly.",
        (f"~{_logop_lbl(us_run_lf)} (US)",
         f"Capacity ~{_logop_lbl(us_cap_lf)} today, riding announced "
         f"megaclusters (Stargate, Hyperion\u2026) to "
         f"~{_logop_num(us_cap_end_lf)} by {horizon.year}. Recent US frontier "
         "models use less than the record run (GPT-5 ~25.8) \u2014 efficiency, "
         "not bigger runs."),
        (f"~{_logop_num(cn_run_lf)} (China)",
         f"Capacity ~{_logop_num(cn_cap_lo_lf)}\u2013"
         f"{_logop_num(cn_cap_hi_lf)} (estimated): plenty of chips (smuggled "
         "NVIDIA + domestic Ascend), but too dispersed and its networking too "
         "export-controlled to fuse into one run."),
        ("the gap widens only slowly",
         f"Capacity grows {10 ** g_us_lo:.1f}\u2013{10 ** g_us_hi:.1f}"
         f"\u00d7/yr (US, measured) vs ~{10 ** g_cn_lo:.1f}\u2013"
         f"{10 ** g_cn_hi:.1f}\u00d7/yr (China). Run points are dated at "
         "estimated training completion (release \u2212 ~1mo) to align with the "
         f"+{run_days // 30}mo capacity line."))

    # Cross-check the growth assumptions against the Data Centers tab's
    # by-country engine, so the two tabs' China stories stay reconciled.
    chk = _cc_country_pace_check(today)
    chk_us, chk_ca, chk_cd = (chk.get(k) for k in
                              (_DC_CTY_US, _DC_CTY_CN_ACCESS, _DC_CTY_CN_DOMESTIC))
    if chk_us or chk_ca or chk_cd:
        bits = []
        if chk_us:
            bits.append(f"US ×{10 ** chk_us['g']:.1f}/yr")
        if chk_cd:
            bits.append(f"China domestic ×{10 ** chk_cd['g']:.1f}/yr")
        if chk_ca:
            bits.append(f"China-accessible ×{10 ** chk_ca['g']:.1f}/yr "
                        "(a from-zero ramp, runs hot)")
        _fn_caption(
            "Cross-checked against the [Data Centers "
            "tab](?tab=datacenters)'s by-country engine.",
            ("Cross-checked", f"That engine fits the catalogued *buildings* at "
                      f"{'; '.join(bits)}. The China band above is a hand-set claim "
                      "about the largest *coherent* run \u2014 fusing dispersed "
                      "chips is the export-controlled step \u2014 kept at or below "
                      "those paces; a refresh that undercuts it forces a retarget."))

    # ── Chart B: ECI derived from compute (Chart A) + shared algorithmic
    # progress — ECI(t) = ECI_now + (a_partial·g_compute + b_algo)·t. The band is
    # each country's compute-growth range; algo is shared, so the divergence is
    # purely the compute gap. ──
    qdates = _cc_quarter_ends(today, horizon)
    x_dates = [today] + qdates
    dt = np.array([(d - today).days / 365.25 for d in x_dates])

    figf = go.Figure()
    _dc_add_projection_band(figf, today, horizon)
    for fr, best, slo, shi, smid, col, rgb, label in (
            (us_fr, us_best, us_eci_slo, us_eci_shi, us_eci_smid,
             '#1F77B4', (31, 119, 180), 'United States'),
            (cn_fr, cn_best, cn_eci_slo, cn_eci_shi, cn_eci_smid,
             '#D62728', (214, 39, 40), 'China')):
        lo, hi, mid = best[1] + slo * dt, best[1] + shi * dt, best[1] + smid * dt
        r, gg, bb = rgb
        figf.add_trace(go.Scatter(
            x=x_dates + x_dates[::-1], y=list(hi) + list(lo[::-1]),
            fill='toself', fillcolor=f'rgba({r},{gg},{bb},0.10)',
            line=dict(width=0), hoverinfo='skip', showlegend=False))
        figf.add_trace(go.Scatter(
            x=[d for d, s, n in fr], y=[s for d, s, n in fr],
            mode='lines+markers', line=dict(color=col, width=1.5),
            marker=dict(size=5, color=col, line=dict(color='white', width=0.5)),
            text=[f"{pretty(n)}<br>ECI {s:.0f}" for d, s, n in fr],
            hoverinfo='text', name=f"{label} (actual)"))
        # Algorithmic-only reference: the same anchor climbing at the *shared*
        # algo rate, with no compute added. The vertical gap up to the full
        # (compute + algo) line is that country's physical-compute contribution —
        # wide for the US, narrow for China, so the gap itself is the compute gap.
        algo_only = best[1] + b_algo * dt
        figf.add_trace(go.Scatter(
            x=x_dates + x_dates[::-1], y=list(mid) + list(algo_only[::-1]),
            fill='toself', fillcolor=f'rgba({r},{gg},{bb},0.16)',
            line=dict(width=0), hoverinfo='skip', showlegend=False))
        figf.add_trace(go.Scatter(
            x=x_dates, y=list(algo_only), mode='lines',
            line=dict(color=col, width=1.3, dash='dot'),
            name=f"{label} (algo only)", hoverinfo='skip'))
        figf.add_trace(go.Scatter(
            x=x_dates, y=list(mid), mode='lines',
            line=dict(color=col, width=2.5, dash='dash'),
            name=f"{label} (compute + algo)", hoverinfo='skip'))
    figf.update_layout(
        height=440, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=55, r=20, t=20, b=40), font=dict(color='#222222'),
        legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)',
                   range=[datetime(2024, 1, 1), horizon],
                   tickfont=dict(color='#222'), title_font=dict(color='#222')),
        yaxis=dict(title_text="Frontier ECI score", gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222'), title_font=dict(color='#222')))
    st.plotly_chart(figf, use_container_width=True)
    _fn_caption(
        f"ECI = <b>~{a_partial:.0f} pts per \u00d710 compute</b> on each "
        f"country's Chart-A compute growth, plus a shared algorithmic term of "
        f"<b>~{b_algo:.0f} pts/yr</b> \u2014 so the divergence is purely the "
        f"compute gap, ~{gap_end:.0f} pts (~{mo_end:.0f} mo) by "
        f"end-{horizon.year}. Dotted = algorithmic-only.",
        ("a shared algorithmic term", "Shared because methods diffuse."),
        ("Dotted = algorithmic-only", "Bands are compute-growth ranges; the "
                                      "shaded gap to the dashed line is "
                                      "compute's contribution, wider for the "
                                      "US."))

    _fn_caption(
        "Order-of-magnitude, not forecasts.",
        ("Order-of-magnitude", "The pooled OLS splits collinear compute and time "
                               "only approximately; US labs under-disclose "
                               "training compute (understating the gap if "
                               "anything); US/China are Epoch's tags, and "
                               "multi-country and untagged models are "
                               "excluded."))

    # ── Where frontier growth comes from: three algorithmic channels ──────────
    # Each channel is measured (or bounded) independently; innovation is the
    # per-country residual against the observed frontier slope, and landing
    # inside the independently derived pure band is the consistency check.
    st.markdown("**Where frontier growth comes from — three algorithmic "
                "channels + compute**")
    isoc_all = _cc_iso_compute(cc_rows)
    _cut = _CC_GAP_WINDOWS[1][1]
    obs_us_fr = _cc_frontier_eci_slope(us_fr, _cut)
    obs_cn_fr = _cc_frontier_eci_slope(cn_fr, _cut)
    if (inno_band and pure_band and isoc_all and isoc_all.get('bands')
            and obs_us_fr and obs_cn_fr):
        top_b = max(isoc_all['bands'], key=lambda b: b['center'])
        dist_prem = max(isoc_all['eci_per_yr'] - top_b['slope'], 0.0)
        diff_mid = max(0.5 * (inno_band[0] + inno_band[1])
                       - 0.5 * (pure_band[0] + pure_band[1]), 0.0)
        comp_us_t = a_partial * 0.5 * (g_us_lo + g_us_hi)
        comp_cn_t = a_partial * 0.5 * (g_cn_lo + g_cn_hi)
        inno_us_r = max(obs_us_fr - comp_us_t - diff_mid, 0.0)
        inno_cn_r = max(obs_cn_fr - comp_cn_t - diff_mid - dist_prem, 0.0)
        st.markdown(
            "| Channel | ECI/yr | How it's measured |\n"
            "|---|---|---|\n"
            f"| **Innovation** (never decays) | {pure_band[0]:.1f}–"
            f"{pure_band[1]:.1f} | pretraining-efficiency prior "
            f"({_CC_PRETRAIN_ALGO_OOM:.1f} OOM/yr × {a_partial:.0f} pts/×10) "
            "up to the ±3 near-frontier refit |\n"
            "| **Diffusion** (methods; dries up ~"
            f"{_CC_DIFF_ABSORB_YRS:.0f} yr after a pause) | ~{diff_mid:.1f} | "
            f"residual: the no-external-teacher level ({inno_band[0]:.1f}–"
            f"{inno_band[1]:.1f}, two convergent near-frontier fits) minus "
            "innovation |\n"
            f"| **Distillation** (decays as the gap closes) | ~{dist_prem:.1f} "
            f"| all-band iso-compute (+{isoc_all['eci_per_yr']:.1f}) minus "
            f"the top band (+{top_b['slope']:.1f}) — only sub-frontier "
            "models have external teachers |")
        figw = go.Figure()
        for name, col, u, c in (
                ('Physical compute', '#D62728', comp_us_t, comp_cn_t),
                ('Innovation', '#1F77B4', inno_us_r, inno_cn_r),
                ('Diffusion', '#6BAED6', diff_mid, diff_mid),
                ('Distillation', '#FF7F0E', 0.0, dist_prem)):
            figw.add_trace(go.Bar(
                y=['China', 'United States'], x=[c, u], orientation='h',
                name=name, marker_color=col,
                hovertemplate=name + ': %{x:.1f} ECI/yr<extra></extra>'))
        for cty, obs in (('United States', obs_us_fr), ('China', obs_cn_fr)):
            figw.add_annotation(x=obs, y=cty, xanchor='left',
                                text=f' = {obs:.1f} observed', showarrow=False,
                                font=dict(size=11, color='#222222'))
        figw.update_layout(
            barmode='stack', height=200, plot_bgcolor='white',
            paper_bgcolor='white', font=dict(color='#222222'),
            margin=dict(l=90, r=90, t=10, b=30),
            legend=dict(orientation='h', y=-0.4, x=0.5, xanchor='center',
                        font=dict(size=11, color='#222')),
            xaxis=dict(title_text="Frontier ECI growth since 2024 (ECI/yr)",
                       gridcolor='rgba(0,0,0,0.12)',
                       tickfont=dict(color='#222'),
                       title_font=dict(color='#222')),
            yaxis=dict(tickfont=dict(color='#222')))
        st.plotly_chart(figw, use_container_width=True)
        _in_band = (pure_band[0] - 0.5 <= inno_us_r <= pure_band[1] + 0.5
                    and pure_band[0] - 0.5 <= inno_cn_r <= pure_band[1] + 0.5)
        _fn_caption(
            "Each bar sums to the country's observed frontier slope since 2024, and "
            "innovation \u2014 the residual \u2014 lands "
            + ("<b>inside</b>" if _in_band else "<b>outside</b>")
            + f" the independently derived {pure_band[0]:.1f}\u2013"
            f"{pure_band[1]:.1f} band for both countries.",
            ("Each bar", "Compute = the exchange rate \u00d7 each side's capacity "
                         "growth. Distillation applies only to the follower (the "
                         "frontier has no stronger teacher); diffusion flows to both "
                         "(methods travel both ways \u2014 MLA/GRPO went "
                         "east-to-west); innovation is the residual."),
            ("the independently derived", "That the residual lands inside a band "
                                          "derived without it is the "
                                          "decomposition's main consistency check."))

    # ── China's algorithmic edge: the distillation scenario ────────────────────
    # Chart B assumes a *shared* algorithmic term — methods diffuse, so both
    # countries gain capability at the same per-year rate at fixed compute. But a
    # compute-constrained follower has both the incentive and the means (distilling
    # from open and API-served frontier models, heavy RL/efficiency focus) to push
    # its own algorithmic rate higher. We test that by measuring each country's
    # algorithmic rate *empirically and separately* — the iso-compute slope (ECI/yr
    # at a fixed compute budget) — instead of forcing one shared term.
    st.markdown("**China's algorithmic edge — the distillation scenario**")
    us_algo, n_us_iso, _ = _cc_iso_compute_rate(cc_rows, 'United States of America')
    cn_algo, n_cn_iso, cn_med = _cc_iso_compute_rate(cc_rows, 'China')
    if us_algo is None or cn_algo is None:
        st.caption("Not enough same-compute-budget models to estimate per-country "
                   "algorithmic rates.")
    else:
        g_cn_mid = 0.5 * (g_cn_lo + g_cn_hi)
        compute_term_cn = a_partial * g_cn_mid          # China's compute-driven ECI/yr
        slope_usalgo = us_algo + compute_term_cn        # China riding US algo rate
        slope_cnalgo = cn_algo + compute_term_cn        # China riding its own algo rate
        premium = cn_algo - us_algo
        premium_pct = (premium / us_algo * 100) if us_algo else 0.0

        # Anchor selector: project forward from today, or backdate to Jan
        # 2025 so the own-rate line doubles as a backtest against China's
        # actual points.
        if st.session_state.get("cc_bd_anchor") not in _CC_BD_OPTIONS:
            st.session_state.pop("cc_bd_anchor", None)
        bd_label = st.selectbox(
            "Scenario lines start from", list(_CC_BD_OPTIONS),
            index=_CC_BD_OPTIONS.index(_CC_DEFAULTS["cc_bd_anchor"]),
            key="cc_bd_anchor",
            help="Today projects forward; Jan 2025 backdates the lines so "
                 "the own-rate one can be checked against China's actual "
                 "frontier since then.")
        if bd_label == "Today":
            anchor_date, cn_anchor, us_anchor = today, cn_best[1], us_best[1]
        else:
            anchor_date = datetime(2025, 1, 1)

            def _fr_at(fr, d):
                vals = [s for dd, s, n in fr if dd <= d]
                return max(vals) if vals else fr[0][1]

            cn_anchor = _fr_at(cn_fr, anchor_date)
            us_anchor = _fr_at(us_fr, anchor_date)
        bd_dates = [anchor_date] + _cc_quarter_ends(anchor_date, horizon)
        dt_bd = np.array([(d - anchor_date).days / 365.25 for d in bd_dates])

        # Own-rate line: distillation decays as the gap closes, the same law as
        # _cc_cn_crossing_sim (gap0 = today's gap) — not the old hard cap. The
        # min() only guards the discrete step against overshoot.
        us_ceiling = us_anchor + us_eci_smid * dt_bd
        cn_traj_us = cn_anchor + slope_usalgo * dt_bd
        gap0_now = max(us_best[1] - cn_best[1], 1e-6)
        inno_mid = 0.5 * (inno_band[0] + inno_band[1]) if inno_band else us_algo
        e = float(cn_anchor)
        cn_traj_cn = [e]
        for i in range(1, len(dt_bd)):
            gap = max(float(us_ceiling[i - 1]) - e, 0.0)
            r = (compute_term_cn + inno_mid
                 + (cn_algo - inno_mid) * min(1.0, gap / gap0_now))
            e = min(e + r * (dt_bd[i] - dt_bd[i - 1]), float(us_ceiling[i]))
            cn_traj_cn.append(e)
        cn_traj_cn = np.array(cn_traj_cn)
        # Two stripped counterfactuals: no *external* distillation (diffusion
        # of methods intact — the inno_band midpoint), and innovation alone
        # (no distillation and no diffusion — the pure band midpoint).
        cn_traj_inno = (cn_anchor + (compute_term_cn + inno_mid) * dt_bd
                        if inno_band else None)
        pure_mid = 0.5 * (pure_band[0] + pure_band[1]) if pure_band else None
        cn_traj_pure = (cn_anchor + (compute_term_cn + pure_mid) * dt_bd
                        if pure_mid is not None else None)

        us_end = float(us_ceiling[-1])
        cn_end_us = float(cn_traj_us[-1])
        cn_end_cn = float(cn_traj_cn[-1])
        mo_us = (us_end - cn_end_us) / us_eci_smid * 12 if us_eci_smid > 0 else float('nan')
        mo_cn = (us_end - cn_end_cn) / us_eci_smid * 12 if us_eci_smid > 0 else float('nan')

        rows_md = [
            "| Scenario | China algo rate | China ECI/yr (algo+compute) | "
            f"China ECI end-{horizon.year} | Behind US |",
            "|---|---|---|---|---|",
            f"| **US-measured algo rate** (mid-compute, itself teacher-fed) | "
            f"{us_algo:.1f} pts/yr | "
            f"{slope_usalgo:.1f} pts/yr | ~{cn_end_us:.0f} | ~{mo_us:.0f} mo |",
            f"| **China's own algo growth** (distillation, decaying) | "
            f"{cn_algo:.1f} pts/yr | {slope_cnalgo:.1f} pts/yr | ~{cn_end_cn:.0f} "
            f"| ~{mo_cn:.0f} mo |"]
        if cn_traj_inno is not None:
            cn_end_inno = float(cn_traj_inno[-1])
            mo_inno = ((us_end - cn_end_inno) / us_eci_smid * 12
                       if us_eci_smid > 0 else float('nan'))
            rows_md.append(
                f"| **No external distillation** (diffusion intact — model "
                f"access cut, papers still flowing) | {inno_mid:.1f} pts/yr | "
                f"{inno_mid + compute_term_cn:.1f} pts/yr | ~{cn_end_inno:.0f} "
                f"| ~{mo_inno:.0f} mo |")
        if cn_traj_pure is not None:
            cn_end_pure = float(cn_traj_pure[-1])
            mo_pure = ((us_end - cn_end_pure) / us_eci_smid * 12
                       if us_eci_smid > 0 else float('nan'))
            rows_md.append(
                f"| **Indigenous only** (no distillation, no diffusion) | "
                f"{pure_mid:.1f} pts/yr | "
                f"{pure_mid + compute_term_cn:.1f} pts/yr | ~{cn_end_pure:.0f} "
                f"| ~{mo_pure:.0f} mo |")
        st.markdown("\n".join(rows_md))

        # Chart: the two China trajectories (backdated to Jan 2025) vs the US
        # frontier ceiling.
        figd = go.Figure()
        _dc_add_projection_band(figd, today, horizon)
        figd.add_trace(go.Scatter(
            x=bd_dates, y=list(us_ceiling), mode='lines',
            line=dict(color='#1F77B4', width=1.5, dash='dot'),
            name='US frontier (ceiling)', hoverinfo='skip'))
        figd.add_trace(go.Scatter(
            x=[d for d, s, n in cn_fr], y=[s for d, s, n in cn_fr],
            mode='lines+markers', line=dict(color='#D62728', width=1.5),
            marker=dict(size=5, color='#D62728', line=dict(color='white', width=0.5)),
            text=[f"{pretty(n)}<br>ECI {s:.0f}" for d, s, n in cn_fr],
            hoverinfo='text', name='China (actual)'))
        figd.add_trace(go.Scatter(
            x=bd_dates, y=list(cn_traj_us), mode='lines',
            line=dict(color='#D62728', width=2.5, dash='dash'),
            name=f'China · US-measured rate ({us_algo:.1f})',
            hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra>US-measured</extra>'))
        figd.add_trace(go.Scatter(
            x=bd_dates, y=list(cn_traj_cn), mode='lines',
            line=dict(color='#7F1010', width=2.5),
            name=f"China · own algo rate ({cn_algo:.1f})",
            hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra>China algo</extra>'))
        if cn_traj_inno is not None:
            figd.add_trace(go.Scatter(
                x=bd_dates, y=list(cn_traj_inno), mode='lines',
                line=dict(color='#6B6B6B', width=2, dash='dashdot'),
                name=f"China · no ext. distillation (~{inno_mid:.0f})",
                hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}'
                              '<extra>no ext. distillation</extra>'))
        if cn_traj_pure is not None:
            figd.add_trace(go.Scatter(
                x=bd_dates, y=list(cn_traj_pure), mode='lines',
                line=dict(color='#9E9E9E', width=2, dash='dot'),
                name=f"China · indigenous only (~{pure_mid:.0f})",
                hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}'
                              '<extra>indigenous</extra>'))
        figd.update_layout(
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=55, r=20, t=20, b=40), font=dict(color='#222222'),
            legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                        bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                        borderwidth=1),
            xaxis=dict(gridcolor='rgba(0,0,0,0.12)',
                       range=[datetime(2024, 1, 1), horizon],
                       tickfont=dict(color='#222'), title_font=dict(color='#222')),
            yaxis=dict(title_text="Frontier ECI score", gridcolor='rgba(0,0,0,0.12)',
                       tickfont=dict(color='#222'), title_font=dict(color='#222')))
        st.plotly_chart(figd, use_container_width=True)
        _fn_caption(
            (f"From today's frontiers (US {us_anchor:.0f}, China {cn_anchor:.0f}; "
             "actual points are history)."
             if bd_label == "Today" else
             f"Backdated to <b>Jan 2025</b> (China \u2248{cn_anchor:.0f} then) so "
             "the edge accumulates over its full period and the own-rate line can "
             "be read against China's actual points.")
            + " The solid line uses China's own measured rate, the dashed line the "
            "US-measured one"
            + ("; the grey dash-dot line cuts model access"
               if cn_traj_inno is not None else "")
            + ("; the dotted grey line strips diffusion too"
               if cn_traj_pure is not None else "")
            + ". The edge buys earlier <i>parity</i>, not a lead, and rests on thin "
            "fits.",
            ("The solid line", "China's own measured algorithmic rate "
                               f"({cn_algo:.1f} vs {us_algo:.1f} ECI/yr, "
                               f"~+{premium_pct:.0f}%), its distillation term "
                               "decaying as the gap closes \u2014 you can't overtake "
                               "your teacher \u2014 so it approaches the US line, "
                               "never past it."),
            ("the dashed line", "Rides the US-measured rate: mid-compute US models, "
                                "themselves teacher-fed \u2014 a follower rate, not "
                                "the US frontier engine. Both China lines share the "
                                f"compute term (~{compute_term_cn:.1f} ECI/yr) and "
                                "differ only algorithmically."),
            *([("the grey dash-dot line", "Cuts model access but keeps method "
                                          f"diffusion (~{inno_mid:.0f} ECI/yr).")]
              if cn_traj_inno is not None else []),
            *([("the dotted grey line", "Strips diffusion too \u2014 innovation "
                                        f"alone (~{pure_mid:.0f} ECI/yr).")]
              if cn_traj_pure is not None else []),
            ("thin fits", f"n={n_us_iso} US / n={n_cn_iso} China within \u00b10.4 "
                          "dex of median compute \u2014 so few same-budget US models "
                          "that the edge is indicative, not measured."))

    # ── When does China cross the target ECI? ─────────────────────────────────
    _render_cc_china_target(
        cn_fr=cn_fr, us_fr=us_fr, a_partial=a_partial, b_algo=b_algo,
        us_algo=us_algo, cn_algo=cn_algo, g_lo=g_cn_lo, g_hi=g_cn_hi,
        us_eci_smid=us_eci_smid, today=today, inno_band=inno_band)


# ── China's ETA to a target ECI ───────────────────────────────────────────
# The level the section asks about: today's US ECI frontier, rounded down to a
# whole point. "When does China cross it" is then "when does China reach where
# the US is now" — a fixed bar, not a moving one, which is why it can be
# answered with a date instead of a gap. The anchor model is deliberately not
# named here: Epoch recomputes scores live, so which model holds the record
# moves between pulls. Retarget when
# TestCcCnTargetIsTodaysUsFrontier fails, and re-read the captions that
# compare this bar with the Pacing pause panel's.
_CC_CN_TARGET_ECI = 169.0


def _cc_cn_target_years(anchor_eci, target, algo_lo, algo_mid, algo_hi,
                        a_partial, g_lo, g_hi, pace_lo=0.85, pace_hi=1.15,
                        release_gap_days=None, n=None):
    """Monte-Carlo years-from-anchor for China's ECI frontier to reach `target`.

    Uses the same two-engine model as Chart B above — a constant algorithmic term
    plus a compute term (`a_partial` ECI per ×10 compute × China's OOM/yr capacity
    growth) — so this section can't quietly disagree with the chart it sits under.
    Each trajectory samples:
      • the algorithmic rate, triangular over the US and China iso-compute fits
        (`algo_mid` is the mode — China's own measured rate is the direct estimate,
        the US rate is the no-distillation-edge alternative);
      • China's compute growth over `[g_lo, g_hi]` (the `_CC_CN_COMPUTE_*` range);
      • an overall pace factor over `[pace_lo, pace_hi]`, mode 1.0. The caller sets
        that range by reality-checking the bottom-up rate against China's *observed*
        frontier slope across the `_CC_GAP_WINDOWS` windows — the two engines are
        estimated off model-level fits and can land above the frontier's actual
        pace, and that disagreement is a real uncertainty the iso-compute spread
        alone (the two fits are within a point of each other) would hide.

    The rate is constant within a trajectory, so the smooth crossing time is
    analytic: (target − anchor) / rate. `release_gap_days` then adds the wait for a
    model to actually ship: the frontier is a step function that only moves on
    releases, so clearing the bar needs a release at or after the smooth crossing.
    Treating releases as roughly Poisson makes that wait Exponential with mean
    equal to the typical inter-release gap.

    Returns (years, rates) as length-n arrays; `years` is NaN wherever the sampled
    rate came out non-positive.
    """
    if n is None:
        n = N_SAMPLES
    lo, hi = min(algo_lo, algo_hi), max(algo_lo, algo_hi)
    if hi - lo < 1e-6:                      # single estimate: give it a little width
        pad = max(0.05 * abs(lo), 0.05)
        lo, hi = lo - pad, hi + pad
    mode = min(max(algo_mid, lo), hi)
    glo, ghi = min(g_lo, g_hi), max(g_lo, g_hi)
    if ghi - glo < 1e-6:
        glo, ghi = glo - 0.01, ghi + 0.01
    plo, phi = min(pace_lo, pace_hi), max(pace_lo, pace_hi)
    if phi - plo < 1e-6:
        plo, phi = plo - 0.05, phi + 0.05
    pmode = min(max(1.0, plo), phi)

    algo = np.random.triangular(lo, mode, hi, n)
    g = np.random.triangular(glo, 0.5 * (glo + ghi), ghi, n)
    pace = np.random.triangular(plo, pmode, phi, n)
    rates = pace * (algo + a_partial * g)
    years = np.where(rates > 0, (target - anchor_eci) / np.where(rates > 0, rates, 1.0),
                     np.nan)
    if release_gap_days and release_gap_days > 0:
        years = years + np.random.exponential(release_gap_days / 365.25, n)
    return years, rates


# The four rate terms of _cc_cn_crossing_sim, in the order they are summed.
_CC_CHANNELS = ('compute', 'innovation', 'diffusion', 'distillation')


def _cc_cn_crossing_sim(anchor_eci, target, *, us_anchor, us_rate, a_partial,
                        g_lo, g_hi, algo_lo, algo_mid, algo_hi,
                        inno_lo, inno_hi, pace_lo, pace_hi,
                        release_gap_days=None, n=None, horizon_yrs=12.0,
                        us_pause_level=None, pure_lo=None, pure_hi=None,
                        t_pause=None, diff_absorb_yrs=_CC_DIFF_ABSORB_YRS,
                        t_dist_stop=None, comp_dead=None, comp_slow=None,
                        dist_teacher=None, channels=None, comp_shadow=None):
    """_cc_cn_target_years with a three-channel algorithmic engine.

        rate = pace · (a_partial·g + pure + diff·D(t)
                       + dist · min(1, gap/gap₀))

    Per sample: `nodist` ~ the [inno_lo, inno_hi] band (the rate with no
    *external* teacher, diffusion intact), `pure` ~ [pure_lo, pure_hi]
    (innovation alone), dist = algo − nodist, diff = nodist − pure. The
    distillation channel decays with the capability gap (you can't overtake
    your teacher; gap vs US(t) = us_anchor + us_rate·t, capped at
    `us_pause_level`); the diffusion channel decays only after a pause —
    D(t) ramps 1→0 over `diff_absorb_yrs` from `t_pause` (years, scalar or
    per-sample), the published stock being absorbed. With `pure_lo=None` or
    `t_pause=None`, diffusion never decays and the law reduces to the
    two-channel model. `t_dist_stop` (years, scalar) cuts the distillation
    channel abruptly at that time — an enforcement action (API-level
    controls), unlike absorption, so no ramp — leaving the other channels
    untouched; None keeps the gap-decay law alone. `comp_dead` (start, end)
    in years zeroes the compute term inside that window — a level setback:
    capacity lost at `start` must be regrown (the window's length = lost
    OOM / regrowth pace) before further growth adds capability again.
    `comp_slow` (t_cut, comp_after) replaces the
    compute term from `t_cut` on — the slower pace left once a channel
    closes — so a cut dated years out doesn't retroactively slow the years
    before it.
    `dist_teacher` (scalar or per-sample) caps the level the distillation
    gap sees: the best *queryable* model when the true frontier is withheld,
    so distillation dries up at the teacher's level, below the bar.

    Pass a dict as `channels` to have it filled with the four rate terms'
    *cumulative* ECI contributions, each an (n, len(grid)) array summing to
    `traj − anchor_eci` — what the Pacing tab's breakdown reads to say which
    channel closed the gap. Off by default: four more arrays of traj's size.
    `comp_shadow` (comp_cap, dead) adds a fifth, `compute_domestic`: the
    compute term recomputed with each sample's own pace capped at `comp_cap`
    and its own `dead` setback window — i.e. what compute would have
    contributed with domestic clusters alone. Capping the *same* sample
    (rather than redrawing) is what makes the difference a clean
    remote-access attribution: when the run already has no access abroad the
    cap never binds and the two coincide exactly. Held ≤ the actual term per
    step, so `compute − compute_domestic` can't go negative while a setback
    is being regrown.

    Returns (years, grid_yrs, traj): crossing years incl. the release wait
    (NaN = not crossed within horizon), the monthly time grid, and the
    (n, len(grid)) capability paths.
    """
    n = n or N_SAMPLES

    def _tri(lo, mid, hi, pad):
        lo, hi = min(lo, hi), max(lo, hi)
        if hi - lo < 1e-6:
            lo, hi = lo - pad, hi + pad
        return np.random.triangular(lo, min(max(mid, lo), hi), hi, n)

    algo = _tri(algo_lo, algo_mid, algo_hi, 0.05)
    g = _tri(g_lo, 0.5 * (g_lo + g_hi), g_hi, 0.01)
    pace = np.maximum(_tri(pace_lo, 1.0, pace_hi, 0.05), 0.0)
    nodist = np.minimum(
        _tri(inno_lo, 0.5 * (inno_lo + inno_hi), inno_hi, 0.05), algo)
    if pure_lo is None:
        pure = nodist
    else:
        pure = np.minimum(
            _tri(pure_lo, 0.5 * (pure_lo + pure_hi), pure_hi, 0.05), nodist)
    dist = algo - nodist
    diffu = nodist - pure
    tp = None if t_pause is None else np.asarray(t_pause)
    comp = a_partial * g
    gap0 = max(us_anchor - anchor_eci, 1e-6)
    dt = 1.0 / 12.0
    steps = int(round(horizon_yrs / dt))
    grid = np.arange(steps + 1) * dt
    traj = np.empty((n, steps + 1))
    traj[:, 0] = anchor_eci
    shadow = channels is not None and comp_shadow is not None
    if channels is not None:
        for _k in _CC_CHANNELS + (('compute_domestic',) if shadow else ()):
            channels[_k] = np.zeros((n, steps + 1))
    e = np.full(n, float(anchor_eci))
    years = np.full(n, np.nan)
    us_rate = np.asarray(us_rate)      # scalar or per-sample array
    for i in range(1, steps + 1):
        us_level = us_anchor + us_rate * grid[i - 1]
        if us_pause_level is not None:
            us_level = np.minimum(us_level, us_pause_level)
        t_level = (us_level if dist_teacher is None
                   else np.minimum(us_level, dist_teacher))
        gap = np.maximum(t_level - e, 0.0)
        d_avail = (1.0 if tp is None else
                   np.clip(1.0 - (grid[i - 1] - tp) / diff_absorb_yrs,
                           0.0, 1.0))
        d_on = 1.0 if (t_dist_stop is None or grid[i - 1] < t_dist_stop) else 0.0
        comp_t = (comp if comp_slow is None or grid[i - 1] < comp_slow[0]
                  else comp_slow[1])
        c_on = (0.0 if comp_dead is not None
                and comp_dead[0] <= grid[i - 1] < comp_dead[1] else 1.0)
        parts = (pace * comp_t * c_on, pace * pure, pace * diffu * d_avail,
                 pace * dist * d_on * np.minimum(1.0, gap / gap0))
        rate = parts[0] + parts[1] + parts[2] + parts[3]
        if channels is not None:
            for _k, _p in zip(_CC_CHANNELS, parts):
                channels[_k][:, i] = channels[_k][:, i - 1] + _p * dt
        if shadow:
            cap, dead = comp_shadow
            d_on_c = (0.0 if dead is not None
                      and dead[0] <= grid[i - 1] < dead[1] else 1.0)
            dom = np.minimum(pace * np.minimum(comp_t, cap) * d_on_c,
                             parts[0])
            channels['compute_domestic'][:, i] = \
                channels['compute_domestic'][:, i - 1] + dom * dt
        e = e + rate * dt
        traj[:, i] = e
        hit = np.isnan(years) & (e >= target)
        if np.any(hit):
            # Interpolate the crossing inside the step: snapping to grid[i]
            # quantizes every crossing to the monthly grid, which swallows
            # sub-month differences between nearby scenarios.
            prev = traj[hit, i - 1]
            tgt = target[hit] if np.ndim(target) else target
            frac = np.clip((tgt - prev) / np.maximum(e[hit] - prev, 1e-12),
                           0.0, 1.0)
            years[hit] = grid[i - 1] + frac * dt
    if release_gap_days and release_gap_days > 0:
        years = years + np.random.exponential(release_gap_days / 365.25, n)
    return years, grid, traj


def _cc_release_gap_days(fr, since=None):
    """Median gap in days between successive steps of a running-max frontier.

    How often the frontier actually moves — i.e. how long a crossing can sit
    waiting on someone to ship. `since` restricts to recent releases (release
    cadence in 2023 says little about 2026). Returns None if too few steps.
    """
    dates = sorted(d for d, s, n in fr if since is None or d >= since)
    if len(dates) < 3:
        return None
    gaps = [(b - a).days for a, b in zip(dates, dates[1:]) if (b - a).days > 0]
    return float(np.median(gaps)) if gaps else None


def _cc_first_reached(fr, target):
    """First (date, name) on a running-max frontier at or above `target`, else None."""
    return next(((d, n) for d, s, n in fr if s >= target), None)


def _wc_region_of(country):
    """The world-share bucket a country falls in — the Data Centers tab's
    regions, which this section shares."""
    r = _dc_region_of(country)
    return r if r in _WC_LABELS else _DC_REGION_OTHER


def _wc_catalogued_shares(dcs, today):
    """Share of *catalogued* H100-equivalents by region, today — the thing the
    published estimates are being used to correct. H100e, not the tab's train
    FLOP: the anchors are stated in H100e."""
    series = _dc_series_for_metric(dcs, 'h100', cap_date=today)
    country_of = {dc['name']: _dc_site_country(dc) for dc in dcs}
    tot = {}
    for name, v in series.items():
        val = _dc_val_at(v['pts'], today)
        if val:
            lab = _wc_region_of(country_of.get(name, ''))
            tot[lab] = tot.get(lab, 0.0) + val
    whole = sum(tot.values())
    if not whole:
        return {}
    return {lab: 100.0 * tot.get(lab, 0.0) / whole for lab in _WC_LABELS}


def _wc_sample_shares(n):
    """Today's world shares, sampled: {region: array of percent} summing to
    100 per sample.

    Each region is lognormal around its central estimate, with the log-spread
    taken from its 80% CI. Samples are renormalized, which pulls the
    medians slightly off the inputs.
    """
    draws = []
    for lab, mid, lo, hi in _WC_REGIONS:
        sigma = np.log(hi / lo) / (2 * 1.2816)
        draws.append(mid * np.exp(np.random.normal(0.0, sigma, n)))
    arr = np.array(draws)
    arr = 100.0 * arr / arr.sum(axis=0)
    return {lab: arr[i] for i, lab in enumerate(_WC_LABELS)}


def _wc_central_shares(years):
    """The plain scenario: {region: percent at each of `years`} with every
    region growing at its central rate. Sums to 100 by construction, and it is
    the line the caption describes, so it is the line drawn — the sampled
    paths supply the range around it."""
    lev = np.array([[mid] for _, mid, _, _ in _WC_REGIONS], dtype=float)
    g = np.array([[mid] for _, mid, _, _ in _WC_GROWTH], dtype=float)
    cur = lev * g ** np.array(years, dtype=float)
    cur = 100.0 * cur / cur.sum(axis=0)
    return {lab: cur[i] for i, lab in enumerate(_WC_LABELS)}


def _wc_share_paths(years, n, common=None):
    """Shares over time: {region: (len(years), n) percent}.

    Every sample grows each region's compute at its own rate from _WC_GROWTH
    and renormalizes, so only the ratios between the rates matter. Rates are
    drawn once per sample and held — this is a trend, not a random walk.

    `common` scales _WC_COMMON_LOAD, the regions' loadings on one global
    shock; 0 makes the rates independent again. Each region's marginal rate
    spread is unchanged either way — only the joint moves, which is what
    decides where a low-US world's compute goes instead.
    """
    w = 1.0 if common is None else common
    base = _wc_sample_shares(n)
    lev = np.array([base[lab] for lab in _WC_LABELS])
    z0 = np.random.normal(0.0, 1.0, n)
    g = []
    for lab, mid, lo, hi in _WC_GROWTH:
        load = min(1.0, w * _WC_COMMON_LOAD[lab])
        g.append(mid * np.exp((np.log(hi / lo) / (2 * 1.2816)) * (
            load * z0
            + np.sqrt(1.0 - load ** 2) * np.random.normal(0.0, 1.0, n))))
    g = np.array(g)
    out = np.empty((len(_WC_LABELS), len(years), n))
    for i, t in enumerate(years):
        cur = lev * g ** t
        out[:, i, :] = 100.0 * cur / cur.sum(axis=0)
    return {lab: out[i] for i, lab in enumerate(_WC_LABELS)}


def _render_cc_china_target(*, cn_fr, us_fr, a_partial, b_algo, us_algo, cn_algo,
                            g_lo, g_hi, us_eci_smid, today, inno_band=None,
                            target=_CC_CN_TARGET_ECI):
    """Section 5: the date China's ECI frontier crosses `target`.

    Everything above reports *gaps* — points behind, months behind, how the gap
    evolves. This turns the same decomposition around and answers the calendar
    question directly, fanning China's compute+algo rate out into crossing
    dates via _cc_cn_crossing_sim (distillation decays as the gap closes);
    _cc_cn_target_years supplies the constant-rate comparison in the caption.
    """
    st.subheader(f"When does China reach ECI {target:.0f} (~Mythos-tier)?")

    cn_best = max(cn_fr, key=lambda x: x[1])
    us_best = max(us_fr, key=lambda x: x[1])
    anchor_d, anchor_eci, anchor_name = cn_best

    if anchor_eci >= target:
        st.success(
            f"**Already there.** {pretty(anchor_name)} reached ECI {anchor_eci:.0f} on "
            f"{anchor_d:%b %-d, %Y}, at or above the {target:.0f} bar.")
        return

    # Algorithmic term: China's own iso-compute rate is the mode, the US rate the
    # no-distillation-edge alternative. Fall back to the shared pooled term when
    # the per-country fits are too sparse to estimate.
    if us_algo is None or cn_algo is None:
        a_lo = a_mid = a_hi = b_algo
        algo_note = (f"the shared pooled algorithmic term (~{b_algo:.0f} ECI/yr); "
                     "per-country iso-compute fits were too sparse")
    else:
        a_lo, a_hi = min(us_algo, cn_algo), max(us_algo, cn_algo)
        a_mid = cn_algo
        algo_note = (f"the iso-compute algorithmic rates ({a_lo:.1f}–{a_hi:.1f} "
                     f"ECI/yr, mode = China's own {cn_algo:.1f})")

    # Reality-check the bottom-up rate against China's *observed* frontier slope
    # over the same windows the gap band uses. The two engines are fitted on
    # model-level data and currently run hot (~14 ECI/yr) against a frontier that
    # has managed 10–13, so the pace factor is widened to span that disagreement
    # rather than asserting a spurious ±12%.
    r_central = a_mid + a_partial * 0.5 * (g_lo + g_hi)
    pace_lo, pace_hi, obs = _cc_cn_pace_band(cn_fr, r_central)
    obs_note = ((f"reality-checked against China's observed frontier slope "
                 f"({min(obs):.1f}–{max(obs):.1f} vs ~{r_central:.1f} ECI/yr "
                 "bottom-up)") if obs else "with a ±15% pace factor")

    # How long a crossing can sit waiting on a release, from the recent cadence.
    gap_d = _cc_release_gap_days(cn_fr, since=today - timedelta(days=730))

    # Distillation-aware headline; the constant-rate model is the comparison.
    inno_lo, inno_hi = inno_band if inno_band else (a_lo, a_hi)
    years, grid_yrs, traj_m = _cc_cn_crossing_sim(
        anchor_eci, target, us_anchor=us_best[1],
        us_rate=max(us_eci_smid, 0.0), a_partial=a_partial,
        g_lo=g_lo, g_hi=g_hi, algo_lo=a_lo, algo_mid=a_mid, algo_hi=a_hi,
        inno_lo=inno_lo, inno_hi=inno_hi,
        pace_lo=pace_lo, pace_hi=pace_hi, release_gap_days=gap_d)
    years_const, rates = _cc_cn_target_years(anchor_eci, target, a_lo, a_mid,
                                             a_hi, a_partial, g_lo, g_hi,
                                             pace_lo=pace_lo, pace_hi=pace_hi,
                                             release_gap_days=gap_d)
    yr_ok = years[np.isfinite(years)]
    if len(yr_ok) < 100:
        st.info("Sampled growth rates were too weak to give a crossing date.")
        return

    def _date_at(y):
        return anchor_d + timedelta(days=float(y) * 365.25)

    y10, y50, y90 = (float(np.percentile(yr_ok, p)) for p in (10, 50, 90))
    d10, d50, d90 = _date_at(y10), _date_at(y50), _date_at(y90)
    rate_med = float(np.median(rates))

    # Lag behind the US at the *same* level: when did (or will) the US frontier
    # first hit this bar? Measured off actual models where one exists, else
    # extrapolated at the US mid rate. Ship-to-ship on BOTH sides: China's
    # crossing keeps its release wait because the US release date embeds the
    # US's own — GPT-5.5 Pro shipped at 161.7 roughly two months after its run
    # finished (Mythos model card), the wait paid in prep and overshoot,
    # invisible in the released steps. A "smooth" variant that interpolated
    # the US steps was tried and read ~1.4 mo low: across a same-day release
    # pair the interpolation collapses to the ship date, charging China no
    # wait while the US date silently carried all of its own.
    us_hit = _cc_first_reached(us_fr, target)
    if us_hit is not None:
        us_hit_d, us_hit_name = us_hit
        us_hit_txt = f"{pretty(us_hit_name)}, {us_hit_d:%b %Y}"
    elif us_eci_smid > 0:
        us_hit_d = us_best[0] + timedelta(
            days=(target - us_best[1]) / us_eci_smid * 365.25)
        us_hit_txt = f"projected {us_hit_d:%b %Y}"
    else:
        us_hit_d, us_hit_txt = None, "—"
    lag_mo = ((d50 - us_hit_d).days / 30.44) if us_hit_d else float('nan')

    m1, m2, m3 = st.columns(3)
    m1.metric(f"China crosses ECI {target:.0f}", f"{d50:%b %Y}",
              f"{d10:%b %Y} – {d90:%b %Y} (80%)", delta_color="off")
    m2.metric("From today", f"~{(d50 - today).days / 30.44:.0f} mo",
              f"from {pretty(anchor_name)} at {anchor_eci:.0f}", delta_color="off")
    # The 80% CI rides under each card's delta line rather than replacing it: the
    # delta says *what the number is measured from* (the anchor model, the US
    # model that set the bar), which the range can't carry. All come off the same
    # d10/d90 as card 1, so the three cards can't disagree.
    m2.caption(f"80% CI: {(d10 - today).days / 30.44:.0f}–"
               f"{(d90 - today).days / 30.44:.0f} mo")
    m3.metric(f"Months behind US when reaching ECI {target:.0f}",
              "—" if us_hit_d is None else f"~{lag_mo:.1f} mo",
              f"US: {us_hit_txt}", delta_color="off")
    if us_hit_d is not None:
        m3.caption(f"80% CI: {(d10 - us_hit_d).days / 30.44:.0f}–"
                   f"{(d90 - us_hit_d).days / 30.44:.0f} mo · shipped model "
                   "to shipped model")

    # ── Chart: China's fan against the fixed target bar ───────────────────────
    # The fan is the smooth *capability* path (rates only). The vertical band and
    # the diamond are the crossing distribution, which also carries the wait for a
    # model to ship — so they sit a little right of where the fan meets the bar.
    # That offset is the release-cadence term, not a plotting error.
    horizon = max(datetime(2027, 12, 31), d90 + timedelta(days=180))
    x_dates = [anchor_d] + _cc_quarter_ends(anchor_d, horizon)
    dt = np.array([(d - anchor_d).days / 365.25 for d in x_dates])
    pct_m = {p: np.percentile(traj_m, p, axis=0) for p in (10, 25, 50, 75, 90)}
    pct = {p: np.interp(dt, grid_yrs, pct_m[p]) for p in pct_m}

    figt = go.Figure()
    _dc_add_projection_band(figt, today, horizon)
    # 80% crossing window as a vertical band, so the answer is readable off the
    # x-axis without tracing the fan up to the target line.
    figt.add_vrect(x0=d10, x1=d90, fillcolor='rgba(214,39,40,0.10)',
                   line_width=0, layer='below')
    # Label the band by hand: plotly can't average a datetime x0/x1 to place its
    # own vrect annotation.
    figt.add_annotation(x=d50, y=1.0, yref='paper', yanchor='bottom',
                        text='80% crossing window', showarrow=False,
                        font=dict(size=10, color='#7F1010'))
    figt.add_hline(y=target, line=dict(color='#111', width=1.5, dash='dash'),
                   annotation_text=f"ECI {target:.0f}",
                   annotation_position='top left',
                   annotation_font=dict(size=11, color='#111'))
    # 0.16 on the outer band, not the 0.10 the other fans use: the eye
    # otherwise reads the inner 50% band as "the fan" and finds it much
    # narrower than the 80% crossing window beside it.
    for lo_p, hi_p, alpha, label in ((10, 90, 0.16, '80% CI'), (25, 75, 0.20, '50% CI')):
        # mode='lines' is load-bearing: this fan spans only ~6 quarters, and under
        # 20 points plotly defaults a Scatter to lines+markers — which would stud
        # the band's outline with stray default-blue dots.
        figt.add_trace(go.Scatter(
            x=x_dates + x_dates[::-1], mode='lines',
            y=list(pct[hi_p]) + list(pct[lo_p][::-1]),
            fill='toself', fillcolor=f'rgba(214,39,40,{alpha})', line=dict(width=0),
            name=label, hoverinfo='skip'))
    # US context: actual frontier plus its mid-rate climb, thin and dotted — the
    # point is that the target bar is already behind the US, not a race to it.
    figt.add_trace(go.Scatter(
        x=[d for d, s, n in us_fr], y=[s for d, s, n in us_fr],
        mode='lines+markers', line=dict(color='#1F77B4', width=1.2),
        marker=dict(size=4, color='#1F77B4', line=dict(color='white', width=0.5)),
        text=[f"{pretty(n)}<br>ECI {s:.0f}" for d, s, n in us_fr],
        hoverinfo='text', name='US (actual)'))
    figt.add_trace(go.Scatter(
        x=x_dates, y=list(us_best[1] + us_eci_smid * dt), mode='lines',
        line=dict(color='#1F77B4', width=1.2, dash='dot'),
        name='US (projected)', hoverinfo='skip'))
    figt.add_trace(go.Scatter(
        x=[d for d, s, n in cn_fr], y=[s for d, s, n in cn_fr],
        mode='lines+markers', line=dict(color='#D62728', width=1.8),
        marker=dict(size=6, color='#D62728', line=dict(color='white', width=0.5)),
        text=[f"{pretty(n)}<br>ECI {s:.0f}" for d, s, n in cn_fr],
        hoverinfo='text', name='China (actual)'))
    figt.add_trace(go.Scatter(
        x=x_dates, y=list(pct[50]), mode='lines',
        line=dict(color='#D62728', width=2.5, dash='dash'),
        name='China (compute + algo)',
        hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra></extra>'))
    figt.add_trace(go.Scatter(
        x=[d50], y=[target], mode='markers',
        marker=dict(size=11, color='#7F1010', symbol='diamond',
                    line=dict(color='white', width=1)),
        name=f'Median crossing ({d50:%b %Y})',
        hovertemplate=f'Median crossing (incl. release wait)<br>{d50:%b %Y}'
                      '<extra></extra>'))
    figt.update_layout(
        height=440, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=55, r=20, t=20, b=40), font=dict(color='#222222'),
        # Bottom-right, unlike the other charts here: both frontiers climb
        # left-to-right, so the usual top-left corner is where the target line and
        # its label live. Under the China line is the one reliably empty corner.
        legend=dict(font=dict(size=11, color='#222'), x=0.99, y=0.02,
                    xanchor='right', yanchor='bottom',
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)',
                   range=[_CC_X_START, horizon],
                   tickfont=dict(color='#222'), title_font=dict(color='#222')),
        yaxis=dict(title_text="Frontier ECI score", gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222'), title_font=dict(color='#222')))
    st.plotly_chart(figt, use_container_width=True)

    # Cumulative odds by quarter — the same distribution read as "have they done it
    # yet?". Quarterly, not annual: the whole band lands inside ~a year, so
    # year-ends would round to 0/100 and say nothing.
    cross_ord = np.array([_date_at(y).toordinal() for y in yr_ok], dtype=float)
    q_cuts = _cc_quarter_ends(today, max(d90 + timedelta(days=95),
                                         today + timedelta(days=370)))
    pmd = ["| Crossed by | Probability |", "|---|---|"]
    for qd in q_cuts:
        p = (cross_ord <= float(qd.toordinal())).mean() * 100
        pmd.append(f"| End {qd.year} Q{(qd.month - 1) // 3 + 1} | **{p:.0f}%** |")
    st.markdown("\n".join(pmd))

    _fn_caption(
        f"China's frontier sits at <b>ECI {anchor_eci:.0f}</b> "
        f"({pretty(anchor_name)}, {anchor_d:%b %Y}), "
        f"<b>{target - anchor_eci:.1f} points</b> short, closing at a median "
        f"<b>~{rate_med:.0f} ECI/yr</b> \u2014 of which compute is the small "
        "term, and the algorithmic part decays as the gap closes."
        + (" A crossing also waits for a release." if gap_d else ""),
        ("closing at a median", f"{algo_note}, plus "
                                f"~{a_partial * 0.5 * (g_lo + g_hi):.1f} ECI/yr "
                                f"from compute ({a_partial:.0f} pts per "
                                f"\u00d710 \u00d7 China's {10 ** g_lo:.1f}"
                                f"\u2013{10 ** g_hi:.1f}\u00d7/yr capacity "
                                f"growth), {obs_note}."),
        ("compute is the small term", "Even doubling China's compute growth "
                                      f"(+{g_hi:.2f} OOM/yr) adds only "
                                      f"~{a_partial * g_hi:.1f} ECI/yr against "
                                      f"~{a_mid:.0f} algorithmic \u2014 moving "
                                      "the date by weeks."),
        ("decays as the gap closes", "The algorithmic term falls toward an "
                                     f"indigenous {inno_lo:.0f}\u2013"
                                     f"{inno_hi:.0f} ECI/yr as China nears the "
                                     "moving US frontier "
                                     f"({(y50 - float(np.nanmedian(years_const))) * 52.18:+.0f}"
                                     " weeks vs constant-rate)."),
        *([("waits for a release", f"One every ~{gap_d:.0f} days lately, an "
                                   "exponential wait \u2014 which is why the "
                                   "diamond sits right of where the fan meets "
                                   "the bar.")] if gap_d else []))
    _fn_caption(
        f"ECI {target:.0f} is a <i>fixed</i> bar \u2014 the US frontier as of "
        f"{us_hit_txt} \u2014 so crossing it means matching where the US is "
        "<b>now</b>, not parity. It is also a bar that can move under a "
        "rescore.",
        ("not parity", "Trajectories climb at a constant rate; only the release "
                       "wait models the frontier's steps, and nothing here "
                       "captures a paradigm shift, a chip shock, or a lab not "
                       "shipping."),
        ("move under a rescore", "Epoch recomputes ECI continuously, and this "
                                 "target sits close to the frontier."))


# ── Per-company buildout-vs-release timing ────────────────────────────────
# Empirical check on the DC→trained→release premise, lab by lab: each lab's
# single largest data center (capacity stepping up as it builds) against its own
# frontier model (a new ECI high). The pipeline predicts a model ~90 days after a
# capacity step (2mo training + ~1mo release lag).

_CC_PANEL_LABS = ["OpenAI", "Anthropic", "Google", "xAI", "Meta"]
_CC_RELEASE_LAG_DAYS = _DAYS_2MO + _CC_RUN_COMPLETION_LAG.days   # 90d, expected release
# Causal eligibility floor for "could this cluster have *trained* the model?" — the
# training run itself (~2mo), without the extra ~1mo release-prep lag. That lag is
# compressible polish, not a hard gate, so a cluster online at least a training run
# before a release can claim it even if the model shipped a few weeks faster than
# the full 90d pipeline (e.g. Sol, 84d after Fairwater Wisconsin). Anything shorter
# than one training run (e.g. a model out ~2 weeks after a DC) still can't.
_CC_TRAIN_FLOOR_DAYS = _DAYS_2MO   # 60d
# Forward-direction counterpart of the floor. "First release on/after step + 90d"
# misses a model that beat the pipeline by a few days — Sol shipped 84d after
# Fairwater Wisconsin, so the strict rule left Wisconsin with no release at all
# while the backward match happily tied Sol to it. A small grace window lets the
# two directions agree. Deliberately *not* the full 30d that the 60d training
# floor would allow: at 30d several clusters start claiming the same, earlier
# model (Meta's Eagle Mountain, Temple and Prometheus steps would all match Muse
# Spark), which makes the forward match degenerate. 7d changes exactly one
# existing match (Google New Albany → Gemini 2.0 Flash, 6d early) and that one
# moves *into* agreement with the backward match.
_CC_EARLY_GRACE_DAYS = 7

# Match a DC site to one of the 5 labs, owner-first for self-built clusters
# (xAI/Meta/Google) then primary-user for labs that rent (OpenAI on
# Microsoft/Oracle, Anthropic on Amazon/Fluidstack). This deliberately differs
# from load_data_centers' generic company_for so Colossus maps to xAI (its owner)
# rather than its listed Anthropic tenant.
# Epoch's owner label for the Colossus sites is SpaceXAI; the releasing lab
# this panel races is xAI. This is the only spelling _DC_COMPANY_ALIASES does
# not already fold (it maps presentation labels, not owner→lab identities).
_CC_LAB_ALIASES = {'SpaceXAI': 'xAI'}


def _cc_lab_for_site(operator, primary_user):
    """The releasing lab a site's buildout belongs to, or None.

    Operator-first for the labs that own their buildings (xAI, Meta, Google),
    then the primary listed tenant — so Colossus 2 stays *xAI's* buildout even
    though Anthropic and Cursor are listed as users. This panel asks whose
    buildout predicts whose releases, so the builder outranks its tenants;
    that is deliberately different from the DC/Pacing tabs' shared-tenancy
    rule, which credits a shared site to every listed tenant.
    """
    op = _CC_LAB_ALIASES.get(operator, operator)
    if op.startswith('Google'):
        return 'Google'
    if op in ('xAI', 'Meta'):
        return op
    u = _CC_LAB_ALIASES.get(primary_user, primary_user)
    if u.startswith('Google'):
        return 'Google'
    if u in _CC_PANEL_LABS:
        return u
    return None


def _dc_meta_mtime():
    p = os.path.join(os.path.dirname(__file__), 'data_centers.csv')
    return os.path.getmtime(p) if os.path.exists(p) else 0


@st.cache_data
def _cc_lab_attribution(_mtime=None):
    """site name → one of the 5 labs (or None), via _cc_lab_for_site.

    Derived from load_data_centers()'s own attribution fields (`operator`,
    `users` — both already aliased through _DC_COMPANY_ALIASES) rather than a
    second read of data_centers.csv, so an Epoch spelling change is handled in
    one place: the loader. Sites whose company label is the site-name fallback
    (`attributed` False) never map to a lab — that token names a landlord, and
    whoever trains there is unknown.
    """
    out = {}
    for dc in dc_all:
        if not dc['attributed']:
            out[dc['name']] = None
            continue
        primary = dc['users'][0] if dc['users'] else ''
        out[dc['name']] = _cc_lab_for_site(dc['operator'], primary)
    return out


def _cc_lab_dc_milestones(lab, attribution, key='perf'):
    """Capacity records of a lab's single largest DC over time: the running-max of
    its sites' chosen capacity metric (`key`), keeping only the dates the record
    steps up. Returns [(date, value, site), …] sorted by date.

    Built on the Data Centers tab's own machinery — _dc_series_for_metric over
    the lab's sites, then _dc_envelope — so "largest site over time" has one
    implementation across the two tabs; only the record-step filter is local.
    """
    sub = [dc for dc in dc_all if attribution.get(dc['name']) == lab]
    series = _dc_series_for_metric(sub, key)
    out = []
    best = 0.0
    for d, val, name, _co in _dc_envelope(series):
        if val is not None and val > best * 1.0001:
            best = val
            out.append((d, val, name))
    return out


# Which ECI Organization labels belong to each lab.
_CC_LAB_ORG_MATCH = {
    'OpenAI': lambda o: o == 'OpenAI',
    'Anthropic': lambda o: o == 'Anthropic',
    'Google': lambda o: o.startswith('Google'),
    'xAI': lambda o: o == 'xAI',
    'Meta': lambda o: o in ('Meta AI', 'Meta'),
}


@st.cache_data
def _cc_company_frontier_models(_mtime=None):
    """Per-lab frontier models (running-max ECI) from the ECI CSV.

    Returns {lab: [(date, eci, name), …]} keeping only releases that set a new
    ECI high for that lab (no global dedup, no date cutoff).
    """
    base = os.path.dirname(__file__)
    rows = list(csv.DictReader(open(os.path.join(base, 'epoch_capabilities_index.csv'))))
    out = {}
    for lab, match in _CC_LAB_ORG_MATCH.items():
        ms = []
        for r in rows:
            if not match((r.get('Organization', '') or '').strip()):
                continue
            sc = (r.get('ECI Score', '') or '').strip()
            ds = (r.get('Release date', '') or '').strip()
            nm = (r.get('Display name', '') or r.get('Model name', '')).strip()
            if not sc or not ds:
                continue
            try:
                sc = float(sc)
                d = datetime.strptime(ds, '%Y-%m-%d')
            except (ValueError, TypeError):
                continue
            ms.append((d, sc, nm))
        ms.sort(key=lambda t: t[0])
        fr = []
        best = -float('inf')
        for d, sc, nm in ms:
            if sc > best:
                best = sc
                fr.append((d, sc, nm))
        out[lab] = fr
    return out


@st.cache_data
def _cc_company_all_releases(_mtime=None):
    """Every release per lab, record-setting or not — the fallback pool.

    A cluster step whose window contains no running-max release still gets a
    row from this list, so a real flagship that Epoch's live rescore pushed
    under its predecessor doesn't leave the step blank. GPT-5.6 Sol is the
    motivating case: the 2026-08-18 pull put it at 161.08, below GPT-5.5 Pro's
    161.73, so it dropped out of _cc_company_frontier_models entirely and
    Fairwater Wisconsin matched nothing.

    Keyed on `(Model name, Release date)`, not `Display name`: the ~10
    reasoning-effort rows of one model share a name and a date and must collapse
    to one release (which suffixed variant wins a dedup is arbitrary and flips
    between Epoch pulls), while a redated revision of the same name is a
    separate release and must not (GPT-4o shipped 2024-05-13 and again
    2024-08-06, and Epoch keeps both). Returns {lab: [(date, eci, name), …]}
    sorted by date then descending ECI, so same-day releases are offered
    strongest first (2026-07-09 ships Sol 161.08, Terra 158.78 and Luna 156.22
    together). Labels are the bare model name; every caller prints the date
    beside it, so two same-named revisions stay distinguishable.
    """
    base = os.path.dirname(__file__)
    rows = list(csv.DictReader(open(os.path.join(base, 'epoch_capabilities_index.csv'))))
    out = {}
    for lab, match in _CC_LAB_ORG_MATCH.items():
        by_name = {}
        for r in rows:
            if not match((r.get('Organization', '') or '').strip()):
                continue
            nm = (r.get('Model name', '') or '').strip()
            ds = (r.get('Release date', '') or '').strip()
            if not nm or not ds:
                continue
            try:
                d = datetime.strptime(ds, '%Y-%m-%d')
            except (ValueError, TypeError):
                continue
            try:
                sc = float((r.get('ECI Score', '') or '').strip())
            except (ValueError, TypeError):
                sc = None
            by_name.setdefault((nm, d), []).append(sc)
        rel = []
        for (nm, d), scores in by_name.items():
            known = [x for x in scores if x is not None]
            rel.append((d, max(known) if known else None, nm))
        out[lab] = sorted(rel, key=lambda t: (t[0], -(t[1] or 0)))
    return out


def _cc_forward_match(step, frontier, all_rel, fm_resp, today):
    """Forward (cluster step → release it enables) match, in three tiers.

    `step` is a `(date, capacity, site)` milestone; `frontier` the lab's
    running-max releases and `all_rel` every release, both `(date, eci, name)`
    sorted by date; `fm_resp` maps a frontier release's `(date, name)` to the
    milestone the *backward* match gave it. Returns `(release, is_fallback)`,
    `(None, False)` when nothing qualifies.

    A step only falls through when the tier above finds nothing:

    1. the first frontier release from `_CC_EARLY_GRACE_DAYS` before the step's
       implied date (step + 90d) onward;
    2. failing that, the earliest frontier release the backward match already
       assigned to this exact step. The backward floor (60d) is looser than the
       forward grace (7d), so a release can be tied to a cluster and still sit
       before its window — Claude Opus 4.8 shipped 24d ahead of New Carlisle's
       implied date. Without this tier the step drops to tier 3 and cites a
       lesser model the backward table doesn't associate with it at all;
    3. failing that, the lab's next release of any kind, so the step renders
       something rather than nothing. Epoch recomputes ECI live, so a real
       flagship can end up scored under its own predecessor and vanish from the
       running max (Sol, 161.08 vs GPT-5.5 Pro's 161.73) — a fact about
       rescoring, not about when the lab shipped, and this panel compares dates
       only. Tier 3 is the only one that reports `is_fallback`; those matches are
       labelled wherever they appear and stay out of the headline median, which
       is a claim about *frontier* releases.

    Tier 3 is skipped for a step that is not yet online: a planned DC has no
    releases to explain.
    """
    pred = step[0] + timedelta(days=_CC_RELEASE_LAG_DAYS)
    floor = pred - timedelta(days=_CC_EARLY_GRACE_DAYS)

    def _first_from(pool):
        return next((t for t in pool if t[0] >= floor), None)

    nm = _first_from(frontier)
    if nm is not None:
        return nm, False
    if step[0] > today:
        return None, False
    own = [t for t in frontier if fm_resp.get((t[0], t[2])) == step]
    if own:
        return min(own, key=lambda t: t[0]), False
    nm = _first_from(all_rel)
    return (nm, True) if nm is not None else (None, False)


# The frontier-model series start in early 2024; before that Epoch's DC tracking
# is too sparse to time releases against.
_CC_PANEL_START = datetime(2024, 1, 1)


def _cc_company_buildout(today, metric_key='perf', kind='sci'):
    """Per-company data-center buildout vs frontier model timing.

    Rendered at the bottom of the Data Centers tab. `metric_key`/`kind` come
    from the tab's capacity-metric selector and drive which metric defines (and
    formats) each "largest DC" capacity step.
    """
    st.subheader("Per-company: does the buildout predict release timing?")
    _fn_caption(
        "A pure timing test: each capacity step of a lab's single largest data "
        f"center, shifted forward {_CC_RELEASE_LAG_DAYS} days to the release "
        "date it implies, against when that lab's models actually shipped. "
        "Capability is ignored. 2024 onward; \u2020 marks a fallback match.",
        (f"{_CC_RELEASE_LAG_DAYS} days", f"{_DAYS_2MO}d training + "
                                         f"{_CC_RUN_COMPLETION_LAG.days}d "
                                         "release lag."),
        ("Capability is ignored", "The only thing compared is *when*, never how "
                                  "good the model is."),
        ("a fallback match", "A cluster whose window holds no record-setting "
                             "release falls back to the lab's next release of "
                             "any kind, rather than showing nothing."))

    lab = st.selectbox("Company", _CC_PANEL_LABS, key="cc_company")
    attribution = _cc_lab_attribution(_mtime=_dc_meta_mtime())
    milestones = _cc_lab_dc_milestones(lab, attribution, key=metric_key)
    fmodels = _cc_company_frontier_models(_mtime=_eci_mtime()).get(lab, [])
    all_rel = _cc_company_all_releases(_mtime=_eci_mtime()).get(lab, [])

    lag = timedelta(days=_CC_RELEASE_LAG_DAYS)
    ms_vis = [m for m in milestones if m[0] >= _CC_PANEL_START]
    fm_vis = [m for m in fmodels if m[0] >= _CC_PANEL_START]
    all_dates = [m[0] + lag for m in ms_vis] + [m[0] for m in fm_vis]
    if not all_dates:
        st.info(f"No 2024-onward data for {lab}.")
        return

    x_end = max(all_dates) + timedelta(days=45)

    # Match *causally*, not by nearest date. Two complementary matches use two
    # different clocks:
    #   • forward  — each DC step → the first frontier release on/after its
    #                *expected* date (step + 90d = train + release-prep lag),
    #                minus a 7d grace so a model that beat the pipeline by a few
    #                days still counts (see _CC_EARLY_GRACE_DAYS).
    #   • backward — each release → the most recent cluster that could have
    #                *trained* it: the latest step online at least one training
    #                run (60d) before the release. The extra ~1mo release lag
    #                isn't required here — a model can ship a few weeks faster
    #                than the full pipeline — but a cluster online less than a
    #                training run before the release still can't claim it.
    # Error = actual − expected (release − (step + 90d)), signed: positive = shipped
    # after the implied date; slightly negative = shipped faster than the pipeline.
    train_floor = timedelta(days=_CC_TRAIN_FLOOR_DAYS)
    act_sorted = sorted(fmodels, key=lambda t: t[0])
    fm_keys = {(t[0], t[2]) for t in fmodels}

    def _responsible_cluster(release):
        cand = [m for m in milestones if (m[0] + train_floor) <= release]
        return cand[-1] if cand else None   # milestones are date-sorted

    # Which cluster the backward match hands each frontier release to.
    # _cc_forward_match consults it so the two directions can't name different
    # releases for the same cluster.
    fm_resp = {(d, n): _responsible_cluster(d) for d, _e, n in fm_vis}

    expected = []
    for step in ms_vis:
        md, mp, mn = step
        pred = md + lag
        nm, fb = _cc_forward_match(step, act_sorted, all_rel, fm_resp, today)
        expected.append({
            'name': mn, 'step': md, 'perf': mp, 'pred': pred,
            'future': md > today, 'model': nm, 'fallback': fb,
            'err': (nm[0] - pred).days if nm else None})

    # Releases shown on the actual-release row: every frontier release, plus any
    # fallback a step pulled in. Both tables and the chart read this one list, so
    # they can't disagree about which releases exist.
    shown = list(fm_vis)
    for e in expected:
        if e['fallback'] and (e['model'][0], e['model'][2]) not in {(d, n) for d, _s, n in shown}:
            shown.append(e['model'])
    shown.sort(key=lambda t: t[0])

    # Per-release backward match (drives the connectors and the recall table).
    model_match = []
    for d, _e, n in shown:
        resp = _responsible_cluster(d)
        pred = (resp[0] + lag) if resp else None
        model_match.append({
            'date': d, 'name': n, 'resp': resp, 'pred': pred,
            'frontier': (d, n) in fm_keys,
            'err': (d - pred).days if pred else None})

    # ── Timeline: predicted release dates (top) vs actual releases (bottom) ──
    Y_PRED, Y_ACT = 1, 0
    fig = go.Figure()
    _dc_add_projection_band(fig, today, x_end)

    # Connector from each release down to the cluster that could have trained it
    # (its implied +90d date), colored by how far — early or late — the release
    # landed from that implied date.
    for m in model_match:
        if m['pred'] is None:
            continue
        ae = abs(m['err'])
        c = '#2CA02C' if ae <= 45 else '#FF7F0E' if ae <= 120 else '#D62728'
        fig.add_trace(go.Scatter(
            x=[m['pred'], m['date']], y=[Y_PRED, Y_ACT], mode='lines',
            line=dict(color=c, width=1.4), hoverinfo='skip', showlegend=False))

    pa = [e for e in expected if not e['future']]
    pf = [e for e in expected if e['future']]
    if pa:
        fig.add_trace(go.Scatter(
            x=[e['pred'] for e in pa], y=[Y_PRED] * len(pa), mode='markers',
            name=f'Predicted (DC + {_CC_RELEASE_LAG_DAYS}d)',
            marker=dict(symbol='diamond', size=11, color='#1F77B4',
                        line=dict(color='white', width=1)),
            hovertext=[f"{e['name']} — {_dc_fmt_value(e['perf'], kind)}<br>"
                       f"online {e['step']:%b %d, %Y} → predicts release "
                       f"~{e['pred']:%b %d, %Y}"
                       + (f"<br>first release after: {e['model'][2]} "
                          f"{e['model'][0]:%b %d, %Y} ({e['err']:+d}d)"
                          + ("<br><i>not an ECI record for this lab</i>"
                             if e['fallback'] else "")
                          if e['model'] else "<br>no release yet")
                       for e in pa],
            hoverinfo='text', showlegend=True))
    if pf:
        fig.add_trace(go.Scatter(
            x=[e['pred'] for e in pf], y=[Y_PRED] * len(pf), mode='markers',
            name='Predicted (planned DC)',
            marker=dict(symbol='diamond-open', size=11, color='#1F77B4',
                        line=dict(color='#1F77B4', width=1.6)),
            hovertext=[f"{e['name']} (planned) — {_dc_fmt_value(e['perf'], kind)}<br>"
                       f"online {e['step']:%b %d, %Y} → predicts release "
                       f"~{e['pred']:%b %d, %Y}" for e in pf],
            hoverinfo='text', showlegend=True))

    def _rel_hover(m):
        return (f"{m['name']}<br>released {m['date']:%b %d, %Y}"
                + ("" if m['frontier'] else
                   "<br><i>not an ECI record for this lab — shown because its "
                   "cluster had no record-setting release</i>")
                + (f"<br>trained on {m['resp'][2]} "
                   f"({_dc_fmt_value(m['resp'][1], kind)}, online "
                   f"{m['resp'][0]:%b %d, %Y})<br>{m['err']:+d}d after implied "
                   f"{m['pred']:%b %d, %Y}"
                   if m['resp'] else "<br>predates any tracked cluster"))

    for is_front, nm, sym in ((True, 'Actual release', 'circle'),
                              (False, 'Release (no ECI record)', 'circle-open')):
        grp = [m for m in model_match if m['frontier'] is is_front]
        if not grp:
            continue
        fig.add_trace(go.Scatter(
            x=[m['date'] for m in grp], y=[Y_ACT] * len(grp),
            mode='markers', name=nm,
            marker=dict(symbol=sym, size=10, color='#2CA02C',
                        line=dict(color='white' if is_front else '#2CA02C',
                                  width=1 if is_front else 1.8)),
            hovertext=[_rel_hover(m) for m in grp],
            hoverinfo='text', showlegend=True))

    fig.update_layout(
        height=300, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=120, r=30, t=20, b=40), font=dict(color='#222222'),
        legend=dict(orientation='h', y=1.18, x=0, font=dict(size=11, color='#222')),
        xaxis=dict(range=[_CC_PANEL_START, x_end], gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222')),
        yaxis=dict(range=[-0.6, 1.6], tickmode='array', tickvals=[Y_ACT, Y_PRED],
                   ticktext=['Actual<br>release', 'Predicted<br>(DC + lag)'],
                   tickfont=dict(color='#222'), showgrid=False, zeroline=False))
    st.plotly_chart(fig, use_container_width=True)

    # ── Verdict line ──
    # Frontier releases only: the sentence is a claim about record-setting
    # models, and the fallbacks are in the panel precisely because they aren't
    # one. They get their own count instead of being averaged in.
    errs = [m['err'] for m in model_match if m['err'] is not None and m['frontier']]
    n_orphan = sum(1 for m in model_match if m['resp'] is None and m['frontier'])
    n_fb = sum(1 for m in model_match if not m['frontier'])
    if not milestones:
        st.info(f"Epoch tracks no {lab} data center, so its releases can't be "
                "timed against a buildout.")
    elif errs:
        med = int(np.median(errs))
        within = sum(1 for x in errs if abs(x) <= 60)
        orphan_note = (f" {n_orphan} release(s) predate any tracked {lab} cluster."
                       if n_orphan else "")
        fb_note = (f" {n_fb} further release(s) shown (hollow) for clusters that "
                   "set no ECI record — not counted here."
                   if n_fb else "")
        _fn_line(
            f"**{lab}: frontier models ship a median {med:+d} days from the date "
            f"their largest cluster's capacity implies** — {within}/{len(errs)} "
            "within 60 days of it."
            + (" Some clusters match nothing." if (orphan_note + fb_note).strip()
               else ""),
            ("their largest cluster's capacity implies",
             f"Capacity online + {_CC_RELEASE_LAG_DAYS}d. Landing within 60 days of "
             "it is capacity-gated; large positive gaps mean the model was limited "
             "by something other than compute."),
            *([("Some clusters match nothing", (orphan_note + fb_note).strip())]
              if (orphan_note + fb_note).strip() else []))

    # The fallback releases are drawn hollow and marked † on the chart; say
    # once why a real flagship can turn up as a non-record.
    if any(e['fallback'] for e in expected):
        st.caption(
            "† not an ECI record for this lab (Epoch rescoring can drop a real "
            "flagship off the running max); shown hollow, excluded from the "
            "median.")

    _fn_caption(
        f"Diamonds = capacity steps + {_CC_RELEASE_LAG_DAYS}d (hollow = "
        "planned) \u00b7 circles = releases \u00b7 connector colour = days off "
        "the implied date (green \u226445, orange \u2264120, red >120). Dates "
        "only, never capability.",
        ("circles = releases", "A release matches only a cluster online "
                               f"\u2265 one training run "
                               f"({_CC_TRAIN_FLOOR_DAYS}d) before it; forward "
                               f"matches allow {_CC_EARLY_GRACE_DAYS}d early."))


def _wc_band(arr):
    """'13% (9–18)' — the median and 80% CI of one region's share."""
    lo, med, hi = (float(np.percentile(arr, q)) for q in (10, 50, 90))
    return f"{med:.0f}% ({lo:.0f}\u2013{hi:.0f})"


def _render_cc_world_shares(today, horizon):
    """Where the world's AI compute is, and where it is heading.

    Every chart above reads Epoch's data-center catalogue, which covers a
    minority of the world's compute and is far denser for the US. This
    section is the outside view: published country estimates for today, then
    each region carried forward at its own growth rate.
    """
    st.subheader("Global compute distribution")

    shares = _wc_sample_shares(N_SAMPLES)
    catalogued = _wc_catalogued_shares(dc_all, today)
    med = {lab: float(np.percentile(a, 50)) for lab, a in shares.items()}
    us, cn = _WC_LABELS[0], _WC_LABELS[1]

    st.markdown(
        f"About **{med[us]:.0f}% of the world's AI compute is in the US** "
        f"today and **{med[cn]:.0f}% in mainland China**. The site catalogue "
        f"the rest of this tab reads puts those at {catalogued.get(us, 0):.0f}% "
        f"and {catalogued.get(cn, 0):.0f}%, but it only tracks large frontier "
        "data centers, and mostly American ones.")

    st.markdown("**Today**")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[med[l] for l in _WC_LABELS], y=list(_WC_LABELS), orientation='h',
        marker=dict(color=[_DC_REGION_COLORS.get(l, "#999999")
                           for l in _WC_LABELS], opacity=0.85),
        error_x=dict(
            type='data', symmetric=False,
            array=[float(np.percentile(shares[l], 90)) - med[l]
                   for l in _WC_LABELS],
            arrayminus=[med[l] - float(np.percentile(shares[l], 10))
                        for l in _WC_LABELS],
            color='#444444', thickness=1.2, width=6),
        name="Best guess", hoverinfo='text',
        hovertext=[f"<b>{l}</b>: {_wc_band(shares[l])}<br>"
                   + "<br>".join(textwrap.wrap(_WC_NOTES[l], 60))
                   for l in _WC_LABELS]))
    fig.add_trace(go.Scatter(
        x=[catalogued.get(l, 0.0) for l in _WC_LABELS], y=list(_WC_LABELS),
        mode='markers',
        marker=dict(symbol='diamond', size=10, color='#FFFFFF',
                    line=dict(color='#222222', width=1.5)),
        name="Tracked data centers only", hoverinfo='text',
        hovertext=[f"{l}: {catalogued.get(l, 0.0):.1f}% of tracked H100e"
                   for l in _WC_LABELS]))
    layout = _dc_layout(False, "", 0, 100, height=300, show_legend=True)
    layout['xaxis'] = dict(layout['xaxis'], title_text="Share of world compute",
                           ticksuffix='%')
    # _dc_layout's y-axis is a value axis; here it holds region names, and a
    # linear type would render the bars against an empty numeric scale.
    layout['yaxis'] = dict(layout['yaxis'], type='category',
                           autorange='reversed', showgrid=False)
    layout['legend'] = dict(layout['legend'], orientation='h', y=-0.3)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Bars are the middle estimate, whiskers the 80% CI \u2014 "
        "which is mostly how much the sources disagree. Each whisker is one "
        "region's own range and they don't add to 100: a world where the US "
        "is at its low end is one where the others are all higher. Diamonds "
        "are what the tracked-site data alone would say. Hover a bar for "
        "where its number comes from. Shares are by location, so Chinese firms' capacity abroad "
        "counts under SEA or Other. Sources: Epoch AI "
        f"([share by country]({_WC_SOURCE_SUPERCOMPUTERS}), "
        f"[smuggling to China]({_WC_SOURCE_SMUGGLING}), "
        f"[its own coverage]({_WC_SOURCE_COVERAGE})), the AI-2027 tracker and "
        "RAND.")

    # ── The same split, carried forward ──
    st.markdown("**Where it is heading**")
    # Opens at the tab's usual start. Before today the same rates simply run
    # backwards — there is no measured history of these shares — and today is
    # in the grid explicitly so the divider lands on a sampled point.
    grid = _dc_cty_month_grid(_CC_X_START, horizon)
    dates = sorted(set([d for d in grid if d < today] + [today]
                       + [d for d in grid if d > today]))
    years = [(d - today).days / 365.25 for d in dates]
    paths = _wc_share_paths(years, N_SAMPLES)
    central = _wc_central_shares(years)
    fig2 = go.Figure()
    for lab in _WC_LABELS:
        color = _DC_REGION_COLORS.get(lab, "#999999")
        band = [f"{np.percentile(paths[lab][i], 10):.0f}\u2013"
                f"{np.percentile(paths[lab][i], 90):.0f}%"
                for i in range(len(dates))]
        fig2.add_trace(go.Scatter(
            x=dates, y=central[lab], name=lab, mode='lines',
            stackgroup='one', line=dict(width=0.5, color=color),
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},"
                      f"{int(color[5:7], 16)},0.75)",
            customdata=band,
            hovertemplate="%{y:.1f}% (80% CI %{customdata})"
                          "<extra>" + lab + "</extra>"))
    # Everything left of today is the rates run backwards, everything right is
    # the projection, and neither is measured — so the past is washed out
    # rather than shaded (a shaded band would sit *under* the filled areas and
    # never show). The divider is heavier than the other tabs' on purpose: it
    # is the only thing separating two halves that otherwise look alike.
    fig2.add_vrect(x0=_CC_X_START, x1=today, fillcolor='#FFFFFF', opacity=0.55,
                   line_width=0, layer='above')
    fig2.add_vline(x=today, line=dict(color='#222222', width=2))
    fig2.add_annotation(x=today, yref='paper', y=1.0, text='<b>Today</b>',
                        showarrow=False, xanchor='right', yanchor='bottom',
                        xshift=-4, font=dict(size=12, color='#222222'))
    fig2.add_annotation(x=dates[-1], yref='paper', y=1.0, text='projected \u2192',
                        showarrow=False, xanchor='right', yanchor='bottom',
                        font=dict(size=10, color='#999999'))
    # Print each end's split in the band itself, so the change is readable
    # without hovering. Thin bands are skipped rather than overlapped.
    for edge, x, anchor, shift in ((dates.index(today), today, 'left', 5),
                                   (-1, dates[-1], 'right', -6)):
        base = 0.0
        for lab in _WC_LABELS:
            v = float(central[lab][edge])
            if v >= 4.0:
                fig2.add_annotation(
                    x=x, y=base + v / 2, text=f"{v:.0f}%", showarrow=False,
                    xanchor=anchor, xshift=shift,
                    font=dict(size=11, color=(
                        '#FFFFFF' if lab in (_WC_LABELS[0], _WC_LABELS[1])
                        else '#222222')))
            base += v
    layout2 = _dc_layout(False, "Share of world compute", _CC_X_START,
                         dates[-1], y_range=[0, 100], show_legend=True,
                         height=360)
    fig2.update_layout(**layout2)
    fig2.update_layout(hovermode='x unified')
    fig2.update_yaxes(ticksuffix='%')
    st.plotly_chart(fig2, use_container_width=True)
    _back = _wc_central_shares([(datetime(2025, 5, 1) - today).days / 365.25])
    st.caption(
        "Anchored on today's split above. There is no measured history of "
        "these shares, so the same rates run backwards before today too "
        f"\u2014 which puts China at {_back[_WC_LABELS[1]][0]:.0f}% in mid-2025, "
        "against Epoch's ~15% reading for May 2025. Bands show each region "
        "growing at its central rate; hover for the 80% CI around that \u2014 "
        "one region at a time, not a joint scenario. Rates: "
        + ", ".join(f"{lab.replace(' domestic', '')} {mid:g}\u00d7/yr"
                    for lab, mid, _, _ in _WC_GROWTH)
        + ". The US and the Gulf are building faster than China is, so "
        "China's share drifts down even as its compute grows.")


def render_compute_capabilities():
    _today = datetime.now()

    with st.sidebar:
        st.header("Compute/capabilities/diffusion")
        include_future = st.checkbox("Include planned future buildout",
                                     value=True, key="cc_future")
        # Run-length options are the Pacing tab's, verbatim.
        if st.session_state.get("cc_run") not in _PC_RUN_OPTIONS:
            st.session_state.pop("cc_run", None)
        run_label = st.selectbox(
            "Run length", list(_PC_RUN_OPTIONS),
            index=list(_PC_RUN_OPTIONS).index(_CC_DEFAULTS["cc_run"]),
            key="cc_run")
        with st.expander("Projection range"):
            cc_end_year = st.radio(
                "Project through", _CC_END_YEARS, horizontal=True,
                index=_CC_END_YEARS.index(_CC_DEFAULTS["cc_end_year"]),
                key="cc_end_year")
        if st.button("Reset", key="cc_reset"):
            for k in _CC_RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state.update(_CC_DEFAULTS)
            st.rerun()

    st.header("Compute/capabilities/diffusion")
    horizon = datetime(cc_end_year, 12, 31)
    run_key = _PC_RUN_OPTIONS[run_label]
    run_days = _DAYS_6MO if run_key == 'train_flop_6mo' else _DAYS_2MO
    run_mo = run_days // 30
    # Shift capacity dates to the "Training run finished" milestone (the DC
    # tab's selector wording): a site online at D has trained a model by D+run.
    # The catalogue cap follows the horizon but never drops below end-2028, so
    # the segment fits (whose eras all end by Jan 2029) are identical whatever
    # year is selected — "Project through" moves the projections, not the rates.
    cap_date = ((datetime(max(2028, cc_end_year), 12, 31) if include_future
                 else _today)
                + timedelta(days=_dc_timing_shift("Training run finished",
                                                  run_days)))
    frontier = _cc_trainflop_frontier(dc_all, cap_date, key=run_key,
                                      run_days=run_days)
    if not frontier:
        st.warning("No data-center data available.")
        return
    # The compute frontier itself is not charted here — its segment fits
    # supply the capacity growth rates the later sections project on, and the
    # captions name the Data Centers tab as their source.
    fits = _cc_segment_fits(frontier, _today)

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: The exchange rate — how much capability per FLOP, and how fast
    # that exchange rate improves (algorithmic efficiency / iso-ECI)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Compute ⟷ ECI")
    cc_rows = load_eci_compute(_mtime=_eci_mtime())
    dec = _cc_decomp(cc_rows)
    eff = _cc_efficiency(cc_rows)
    if dec is None or eff is None:
        st.warning("Not enough models with both ECI and training-compute data.")
        return

    st.markdown(
        "| Exchange rate | Value | What it means |\n"
        "|---|---|---|\n"
        f"| Compute → capability | **+{eff['eci_per_oom']:.0f} ECI** per 10× compute "
        "| at a fixed moment in time |\n"
        f"| Capability gets cheaper | **−{eff['g_central']:.1f} OOM/yr** "
        f"(÷{eff['algo_mult']:.1f}/yr) | hold ECI fixed → less compute needed |")
    _fn_caption(
        f"Based on {dec['n']} models reporting training compute. Row 2 is "
        "algorithmic efficiency.",
        ("algorithmic efficiency", "Better architectures, data, RL, "
                                   "post-training, scaffolding \u2014 i.e. "
                                   "\u201ceffective compute per real "
                                   "operation\u201d going up."))

    # Iso-ECI scatter: compute vs date. Continuous ECI-by-color reads poorly, so
    # we use discrete capability bands — each band's dots and its downward fit
    # line share a distinct color; models outside any band are grey context.
    _BAND_COLORS = {105: '#4C78A8', 115: '#F58518', 125: '#54A24B'}
    band_centers = {b['center'] for b in eff['bands']}

    def _in_band(m, c):
        return abs(m['eci'] - c) <= _CC_BAND_HALFWIDTH

    figx = go.Figure()
    other = [m for m in cc_rows if not any(_in_band(m, c) for c in band_centers)]
    figx.add_trace(go.Scatter(
        x=[m['date'] for m in other], y=[10.0 ** m['log10_flop'] for m in other],
        mode='markers', marker=dict(size=5, color='#D9D9D9', line=dict(width=0)),
        text=[f"{m['name']}<br>ECI {m['eci']:.0f}" for m in other],
        hoverinfo='text', name='outside bands', showlegend=True))
    for bseg in eff['bands']:
        c = bseg['center']
        col = _BAND_COLORS.get(c, '#D62728')
        rate = 10 ** (-bseg['slope'])
        mem = [m for m in cc_rows if _in_band(m, c)]
        figx.add_trace(go.Scatter(
            x=[m['date'] for m in mem], y=[10.0 ** m['log10_flop'] for m in mem],
            mode='markers',
            marker=dict(size=7, color=col, line=dict(color='white', width=0.5)),
            text=[f"{m['name']}<br>ECI {m['eci']:.0f}" for m in mem],
            hoverinfo='text', legendgroup=str(c),
            name=f"ECI {c}±{_CC_BAND_HALFWIDTH:.0f}  →  compute ÷{rate:.1f}/yr",
            showlegend=True))
        figx.add_trace(go.Scatter(
            x=bseg['fit_x'], y=[10.0 ** v for v in bseg['fit_y']], mode='lines',
            line=dict(color=col, width=2.5, dash='dot'),
            legendgroup=str(c), hoverinfo='skip', showlegend=False))
    figx.update_layout(
        height=440, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=70, r=20, t=10, b=40), font=dict(color='#222222'),
        legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222'),
                   title_font=dict(color='#222')),
        yaxis=dict(gridcolor='rgba(0,0,0,0.12)'))
    _cc_logop_yaxis(figx, "Training compute (log₁₀ OP)")
    st.plotly_chart(figx, use_container_width=True)
    _fn_caption(
        "The compute needed to stay in a band slopes <i>down</i> over time "
        "\u2014 that downward slope is the efficiency rate.",
        ("a band", "Each colored set is a fixed-capability band (ECI \u00b1 "
                   f"{_CC_BAND_HALFWIDTH:.0f}); its matching dotted line is the "
                   "within-band fit. Grey dots sit outside these bands."))

    # Time-to-cheaper table.
    st.markdown("**Time to reach the same ECI at less compute** "
                f"(central ≈{eff['g_central']:.1f} OOM/yr; range "
                f"{eff['g_lo']:.1f}–{eff['g_hi']:.1f} from the band vs all-data fits):")
    tmd = ["| Compute reduction | Time to match capability | Range |",
           "|---|---|---|"]
    for f in (2, 5, 10):
        tm = eff['times'][f]
        tmd.append(f"| **{f}× less** | ~{tm['central']:.0f} months | "
                   f"{tm['lo']:.0f}–{tm['hi']:.0f} mo |")
    st.markdown("\n".join(tmd))
    _fn_caption(
        f"Inverse regression gives \u2212\u03b2\u209c = {eff['g_inv']:.2f} "
        f"OOM/yr (R\u00b2 {eff['r2']:.2f}); the iso-ECI bands give a median "
        f"{eff['band_median']:.2f} OOM/yr. This is total capability efficiency.",
        ("Inverse regression", "log10(OP) = \u03b1\u00b7ECI + "
                               "\u03b2\u209c\u00b7t + c."),
        ("total capability efficiency", "ECI rewards reasoning/RL/post-training, "
                                        "so this runs faster than "
                                        "pure-pretraining algorithmic "
                                        "efficiency."))

    # Mirror image: hold compute fixed, watch ECI climb.
    isoc = _cc_iso_compute(cc_rows)
    if isoc is not None:
        _fn_line(
            "**The mirror image — same compute, rising capability.** A model "
            f"trained on the same compute a year later scores about "
            f"**+{isoc['eci_per_yr']:.0f} ECI points** higher.",
            ("The mirror image", "Same engine as the chart above, axes flipped: "
                                 "hold the compute budget fixed and watch ECI "
                                 "climb."),
            ("higher", f"Range {isoc['lo']:.0f}\u2013{isoc['hi']:.0f} across "
                       "budgets."))
        _CBAND_COLORS = {23.5: '#8C6BB1', 24.5: '#3690C0', 25.5: '#02818A'}

        def _in_cband(m, c):
            return abs(m['log10_flop'] - c) <= _CC_CBAND_HALFWIDTH

        cband_centers = {b['center'] for b in isoc['bands']}
        figc = go.Figure()
        cother = [m for m in cc_rows
                  if not any(_in_cband(m, c) for c in cband_centers)]
        figc.add_trace(go.Scatter(
            x=[m['date'] for m in cother], y=[m['eci'] for m in cother],
            mode='markers', marker=dict(size=5, color='#D9D9D9', line=dict(width=0)),
            text=[f"{m['name']}<br>ECI {m['eci']:.0f}" for m in cother],
            hoverinfo='text', name='outside bands', showlegend=True))
        for bseg in isoc['bands']:
            c = bseg['center']
            col = _CBAND_COLORS.get(c, '#02818A')
            mem = [m for m in cc_rows if _in_cband(m, c)]
            figc.add_trace(go.Scatter(
                x=[m['date'] for m in mem], y=[m['eci'] for m in mem],
                mode='markers',
                marker=dict(size=7, color=col, line=dict(color='white', width=0.5)),
                text=[f"{m['name']}<br>ECI {m['eci']:.0f}" for m in mem],
                hoverinfo='text', legendgroup=str(c),
                name=f"~{_logop_num(c)} log OP  →  +{bseg['slope']:.0f} ECI/yr",
                showlegend=True))
            figc.add_trace(go.Scatter(
                x=bseg['fit_x'], y=bseg['fit_y'], mode='lines',
                line=dict(color=col, width=2.5, dash='dot'),
                legendgroup=str(c), hoverinfo='skip', showlegend=False))
        figc.update_layout(
            height=420, plot_bgcolor='white', paper_bgcolor='white',
            margin=dict(l=55, r=20, t=10, b=40), font=dict(color='#222222'),
            legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                        bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                        borderwidth=1),
            xaxis=dict(gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222'),
                       title_font=dict(color='#222')),
            yaxis=dict(title_text="ECI score", gridcolor='rgba(0,0,0,0.12)',
                       tickfont=dict(color='#222'), title_font=dict(color='#222')))
        st.plotly_chart(figc, use_container_width=True)
        _fn_caption(
            "Each dotted line slopes <i>up</i> \u2014 that's ECI gained per year at "
            "a constant compute budget.",
            ("Each dotted line", "Each colored set is a fixed compute band "
                                 f"(log\u2081\u2080 OP \u00b1 "
                                 f"{_CC_CBAND_HALFWIDTH:.1f} dex). Same engine as "
                                 "the chart above, axes flipped."))

        # Mirror of the time-to-cheaper table: hold the compute budget fixed and
        # read off the ECI gained over time. The last column converts that gain to
        # the compute multiplier it's worth (via the exchange rate), tying this
        # table back to the one above.
        st.markdown("**ECI gained at a fixed compute budget** "
                    f"(central ≈{isoc['eci_per_yr']:.0f} ECI/yr; range "
                    f"{isoc['lo']:.0f}–{isoc['hi']:.0f} across budgets):")
        epo = eff['eci_per_oom']
        imd = ["| Time at same compute | ECI gained | Range | Worth ~ |",
               "|---|---|---|---|"]
        for yrs in (1, 2, 3):
            c = isoc['eci_per_yr'] * yrs
            lo, hi = isoc['lo'] * yrs, isoc['hi'] * yrs
            oom = c / epo if epo else float('nan')
            imd.append(f"| **{yrs} year{'s' if yrs > 1 else ''}** | "
                       f"+{c:.0f} ECI | +{lo:.0f} to +{hi:.0f} | "
                       f"{10 ** oom:.0f}× more compute |")
        st.markdown("\n".join(imd))
        _fn_caption(
            f"At a constant budget, a year's algorithmic progress adds "
            f"~{isoc['eci_per_yr']:.0f} ECI \u2014 the same capability you'd "
            f"otherwise have to buy with "
            f"~{10 ** (isoc['eci_per_yr'] / epo):.0f}\u00d7 more compute.",
            ("more compute", f"At {epo:.0f} ECI per \u00d710 compute. The mirror "
                             "image of the table above: there capability gets "
                             "cheaper, here the same spend buys more."))

    # Two engines — what a compute slowdown really costs. Flows on from the
    # exchange-rate section above (no separate header).
    eci_per_oom = eff['eci_per_oom']
    g_frontier = dec['frontier_compute_oom']          # frontier-model physical OOM/yr
    obs_slope = dec['eci_frontier_slope']             # observed frontier ECI pts/yr
    g_recent = fits[-2]['slope_oom'] if len(fits) >= 2 else g_frontier   # capacity now
    g_planned = fits[-1]['slope_oom'] if len(fits) >= 1 else g_recent    # capacity planned

    # Algorithmic efficiency is bracketed by the two views of Section 2, which
    # disagree by ~2× (regression dilution). Low end = iso-ECI (hold capability,
    # compute falls); high end = iso-compute (hold compute, ECI rises) converted
    # to OOM/yr via the neutral (geometric-mean) exchange rate. The truth sits
    # between; we report the band and use the geometric mean as central.
    g_algo_lo = eff['g_central']                      # iso-ECI family, OOM/yr
    xr_neutral = (dec['a_partial'] * eci_per_oom) ** 0.5 if dec['a_partial'] > 0 else eci_per_oom
    g_algo_hi = (isoc['eci_per_yr'] / xr_neutral) if (isoc and xr_neutral > 0) else g_algo_lo
    if g_algo_hi < g_algo_lo:
        g_algo_lo, g_algo_hi = g_algo_hi, g_algo_lo
    g_algo_mid = (g_algo_lo * g_algo_hi) ** 0.5

    def _phys_share(g_algo):
        return g_recent / (g_recent + g_algo) if (g_recent + g_algo) else 0.0

    share_hi = _phys_share(g_algo_lo)                 # compute-favorable
    share_mid = _phys_share(g_algo_mid)
    share_lo = _phys_share(g_algo_hi)                 # algo-favorable

    st.markdown(
        f"#### ~{obs_slope:.0f} ECI/yr  =  physical compute  +  algorithmic "
        "efficiency  →  effective compute")
    st.caption(
        "The two views above (compute-constant and ECI-constant) disagree by ~2× "
        "due to regression dilution, so each engine below is given as a range.")
    phys_lo, phys_hi = obs_slope * share_lo, obs_slope * share_hi
    algo_lo, algo_hi = obs_slope * (1 - share_hi), obs_slope * (1 - share_lo)
    e1, e2, e3 = st.columns(3)
    e1.metric("Physical compute", f"~{phys_lo:.0f}–{phys_hi:.0f} ECI/yr",
              f"×{10**g_recent:.1f}/yr capacity")
    e2.metric("Algorithmic efficiency", f"~{algo_lo:.0f}–{algo_hi:.0f} ECI/yr",
              "iso-ECI ↔ iso-compute")
    e3.metric("Share of growth of compute", f"{share_lo*100:.0f}–{share_hi*100:.0f}%",
              "≈ ⅓ to ½")
    _fn_caption(
        f"The \u00d7{10**g_recent:.1f}/yr capacity pace comes from the [Data "
        "Centers tab](?tab=datacenters), on lab-attributable sites only.",
        ("lab-attributable sites only",
         f"The \u201c{fits[-2]['label'] if len(fits) >= 2 else ''}\u201d fit of "
         "the largest-lab-site train-FLOP series, restricted to sites "
         "attributable to a model-shipping lab \u2014 neutral hosts excluded, "
         "deliberately stricter than that tab's own record line."))
    st.markdown(
        f"So **physical compute drives roughly a third to a half** of the "
        f"~{obs_slope:.0f} ECI-points/yr.")

    # Distillation control: the same regression on frontier-grade models only —
    # near-frontier ECI at frontier-scale compute, the subset that could not
    # lean on a stronger teacher.
    fg = _cc_frontier_grade_algo(
        cc_rows, load_eci_frontier(_mtime=_eci_mtime(), full_window=True))
    share_fg = None
    if fg is not None:
        xr_fg = ((fg['a_partial'] * eci_per_oom) ** 0.5
                 if fg['a_partial'] > 0 else xr_neutral)
        if xr_fg > 0:
            share_fg = _phys_share(fg['b_time'] / xr_fg)
            _lvl = _cc_cn_level_offset(cc_rows)
            _lvl_note = (
                f" China's edge is a level, not a rate: at matched compute and "
                f"date its models sit ~{_lvl[0]:+.0f} ECI above US peers "
                f"(n={_lvl[1]}), while the two countries' iso-compute slopes "
                "are indistinguishable." if _lvl else "")
            _fn_caption(
                f"<b>Distillation control:</b> refitting on frontier-grade models "
                f"slows the time coefficient, so at the frontier the algo engine is "
                f"smaller and compute's share is nearer ~{share_fg * 100:.0f}% even on "
                "the algo-favorable estimator.",
                ("frontier-grade models",
                 f"Within 5 ECI of the running frontier at release AND trained "
                 f"within {_CC_FG_FLOP_MARGIN:.0f} OOM of the frontier run, "
                 f"n={fg['n']}. The compute screen matters: near-frontier ECI at "
                 "10\u2013100\u00d7 less compute is the distillation fingerprint "
                 "itself (DeepSeek, Qwen, Kimi), so a capability margin alone "
                 "admits the heaviest distillers to the control." + _lvl_note),
                ("slows the time coefficient",
                 f"+{fg['b_time']:.1f} ECI/yr at fixed compute vs "
                 f"+{dec['b_time']:.1f} all-model \u2014 followers ride a teacher, "
                 "the frontier cannot. The compute coefficient does <i>not</i> "
                 f"steepen (+{fg['a_partial']:.1f} vs +{dec['a_partial']:.1f} per "
                 "\u00d710): reasoning-era models reach the frontier at "
                 "sub-frontier compute."))

    # One 100%-of-growth split bar. The boundary between the two engines isn't
    # pinned down, so the contested middle (iso-compute share → iso-ECI share) is
    # drawn as a hatched band, with a solid line marking the central-view split.
    algo_min = (1 - share_hi) * 100          # guaranteed-algorithmic share
    contested = (share_hi - share_lo) * 100  # boundary could lie anywhere here
    phys_min = share_lo * 100                # guaranteed-physical share
    central_split = (1 - share_mid) * 100    # central-view boundary (algo share)
    figs = go.Figure()
    figs.add_trace(go.Bar(
        y=[''], x=[algo_min], orientation='h', name='Algorithmic efficiency',
        marker_color='#1F77B4', hovertemplate='Algorithmic ≥%{x:.0f}%<extra></extra>'))
    figs.add_trace(go.Bar(
        y=[''], x=[contested], orientation='h', name='contested boundary',
        marker=dict(color='#ECECEC',
                    pattern=dict(shape='/', fgcolor='#9AA0A6', size=8, solidity=0.35)),
        hovertemplate='either engine: %{x:.0f}%<extra></extra>'))
    figs.add_trace(go.Bar(
        y=[''], x=[phys_min], orientation='h', name='Physical compute',
        marker_color='#D62728', hovertemplate='Physical ≥%{x:.0f}%<extra></extra>'))
    figs.add_annotation(x=algo_min / 2, y=0, text='Algorithmic', showarrow=False,
                        font=dict(color='white', size=13))
    figs.add_annotation(x=algo_min + contested + phys_min / 2, y=0, text='Physical',
                        showarrow=False, font=dict(color='white', size=13))
    figs.add_vline(x=central_split, line=dict(color='#111', width=2.5),
                   annotation_text=f'central view (~{share_mid*100:.0f}% physical)',
                   annotation_position='top', annotation_yshift=2,
                   annotation_font=dict(size=10, color='#111'))
    if share_fg is not None:
        figs.add_vline(x=(1 - share_fg) * 100,
                       line=dict(color='#555555', width=1.8, dash='dash'),
                       annotation_text=f'frontier-grade (~{share_fg*100:.0f}%)',
                       annotation_position='bottom',
                       annotation_font=dict(size=10, color='#555555'))
    figs.update_layout(
        barmode='stack', height=190, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=28, b=30), font=dict(color='#222222'),
        legend=dict(orientation='h', y=-0.55, x=0.5, xanchor='center',
                    font=dict(size=11, color='#222')),
        xaxis=dict(title_text="Share of frontier ECI growth", range=[0, 100],
                   ticksuffix='%', gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222'), title_font=dict(color='#222')),
        yaxis=dict(showticklabels=False))
    st.plotly_chart(figs, use_container_width=True)
    _fn_caption(
        f"<b>Hatched band</b> = physical compute's contested share, "
        f"<b>{share_lo*100:.0f}\u2013{share_hi*100:.0f}%</b>; <b>solid "
        f"line</b> = central view (~{share_mid*100:.0f}%). Algorithms are the "
        "rest.",
        ("contested share", "Low = iso-compute fit, high = iso-ECI fit. "
                            "Frontier-model compute growth "
                            f"(\u00d7{10**g_frontier:.1f}/yr) argues for the "
                            "high end."))

    _top_band = (max(isoc['bands'], key=lambda b: b['center'])
                 if isoc and isoc.get('bands') else None)
    # st.warning takes no HTML, so the caveats render as a flagged markdown
    # line rather than a yellow box — the hovers are worth more than the box.
    _fn_line(
        "\u26a0\ufe0f **Caveats.** Order-of-magnitude, not forecasts. The two "
        "engines aren't independent; ECI is not pretraining efficiency; "
        "cheap-model data is sparse; and the capacity series is a ceiling"
        + ("; the iso-compute rates include distillation." if _top_band else "."),
        ("aren't independent", "Algorithmic progress is compute-fed, so a "
                               "physical slowdown drags the algorithmic rate "
                               "too, making a stall worse than the "
                               "\u2153\u2013\u00bd shown."),
        ("not pretraining efficiency", "ECI bundles post-training/RL, so this is "
                                       "total-capability efficiency."),
        ("cheap-model data is sparse", "Labs rarely retrain small models to "
                                       "re-hit old levels (Qwen, Kimi, "
                                       "distilled MoEs)."),
        ("a ceiling", f"The {run_mo}mo-capacity series is a ceiling, not "
                      "per-model training compute."),
        *([("include distillation", "Which the frontier itself cannot use "
                                    "\u2014 the top compute band runs "
                                    f"+{_top_band['slope']:.0f} vs "
                                    f"+{isoc['eci_per_yr']:.0f} ECI/yr central. "
                                    "The frontier-grade refit above is the "
                                    "control.")] if _top_band else []))
    _fn_caption(
        "Data: Epoch AI Capabilities Index + Frontier Data Centers. The "
        "efficiency band spans two OLS directions.",
        ("two OLS directions", "Iso-ECI (compute on ECI+time) and iso-compute "
                               "(ECI on compute+time), which bracket the true "
                               "rate (errors-in-variables dilution). Frontier "
                               "rates use the running-max-ECI subset."))

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: ECI Forecasts — quarterly frontier projection to end of 2029
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("ECI Forecasts")
    _cc_eci_forecast(cc_rows, frontier, _today, obs_slope, g_recent, g_planned,
                     share_lo, share_mid, share_hi, horizon=horizon)

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: US vs. China — the same decomposition read by country
    # ══════════════════════════════════════════════════════════════════════
    _cc_us_vs_china(cc_rows, _today, horizon=horizon,
                    run_key=run_key, run_days=run_days)

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: the world split the catalogue above cannot see
    # ══════════════════════════════════════════════════════════════════════
    _render_cc_world_shares(_today, horizon)


# ── URL parameter persistence ────────────────────────────────────────────

_REV_DEFAULTS = {
    "rev_end_year": 2026,
    "rev_log_scale": True,
    "rev_2025_only": False,
    "rev_milestones": True,
    "rev_labels": True,
    "rev_combined": False,
}
_REV_TRACKED_KEYS = list(_REV_DEFAULTS.keys()) + [
    "rev_proj_as_of", "oai_n_recent", "ant_n_recent", "comb_n_recent",
    "oai_dt_lo", "oai_dt_hi", "ant_dt_lo", "ant_dt_hi",
    "comb_dt_lo", "comb_dt_hi",
]

_ECG_DEFAULTS = {"ecg_highlight": "None"}
_ECG_TRACKED_KEYS = list(_ECG_DEFAULTS.keys())

# Internal cache keys: never round-trip through URL
_URL_EXCLUDED_SUFFIXES = ("_seg_config",)

def _all_tracked():
    """Return (ordered keys, combined defaults) across every tab."""
    keys = []
    defaults = {}
    for ks, ds in [
        (_METR_RESET_KEYS, _METR_DEFAULTS),
        (_eci_tab_reset_keys("eci"), _eci_tab_defaults("eci")),
        (_eci_tab_reset_keys("ecicn"), _eci_tab_defaults("ecicn")),
        (_RLI_RESET_KEYS, _RLI_DEFAULTS),
        (_RSI_RESET_KEYS, _RSI_DEFAULTS),
        (_UKC_RESET_KEYS, _UKC_DEFAULTS),
        (_EMP_RESET_KEYS, _EMP_DEFAULTS),
        (_REV_TRACKED_KEYS, _REV_DEFAULTS),
        (_ECG_TRACKED_KEYS, _ECG_DEFAULTS),
        (_DC_RESET_KEYS, _DC_DEFAULTS),
        (_CC_RESET_KEYS, _CC_DEFAULTS),
        (_PC_RESET_KEYS, _PC_DEFAULTS),
    ]:
        keys.extend(ks)
        defaults.update(ds)
    # Dedupe while preserving order
    seen = set()
    deduped = []
    for k in keys:
        if k in seen or any(k.endswith(s) for s in _URL_EXCLUDED_SUFFIXES):
            continue
        seen.add(k)
        deduped.append(k)
    return deduped, defaults

def _coerce_url_value(raw, default):
    if isinstance(default, bool):
        return raw in ("1", "true", "True")
    if isinstance(default, int):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default
    return str(raw)

def _coerce_unknown_url_value(raw):
    s = str(raw)
    if s.lstrip("-").isdigit():
        try:
            return int(s)
        except ValueError:
            pass
    try:
        return float(s)
    except ValueError:
        return s

def _hydrate_session_from_url():
    keys, defaults = _all_tracked()
    qp = st.query_params
    # Remember which keys came from URL so we don't poison the baseline snapshot with them
    if "_url_keys_at_load" not in st.session_state:
        st.session_state["_url_keys_at_load"] = set(qp.keys())
    for k in keys:
        if k in st.session_state:
            continue
        if k not in qp:
            continue
        raw = qp[k]
        if k in defaults:
            st.session_state[k] = _coerce_url_value(raw, defaults[k])
        else:
            st.session_state[k] = _coerce_unknown_url_value(raw)

_BASELINE_SENTINEL = object()

def _sync_session_to_url():
    keys, defaults = _all_tracked()
    qp = st.query_params
    if "_url_baseline" not in st.session_state:
        st.session_state["_url_baseline"] = {}
    baseline = st.session_state["_url_baseline"]
    url_keys_at_load = st.session_state.get("_url_keys_at_load", set())
    for k in keys:
        if k not in st.session_state:
            if k in qp:
                del qp[k]
            continue
        val = st.session_state[k]
        # Capture baseline for keys whose initial value wasn't supplied via URL
        if k not in baseline and k not in url_keys_at_load and k not in defaults:
            baseline[k] = val
        if k in defaults:
            eff_default = defaults[k]
        elif k in baseline:
            eff_default = baseline[k]
        else:
            eff_default = _BASELINE_SENTINEL
        if val == eff_default:
            if k in qp:
                del qp[k]
            continue
        if isinstance(val, bool):
            qp[k] = "1" if val else "0"
        else:
            qp[k] = str(val)



# ══════════════════════════════════════════════════════════════════════════
# Pacing — when does each entity first command a threshold-scale run?
# ══════════════════════════════════════════════════════════════════════════

# The bar a pacing agreement (or an observer) might care about, in total 8-bit
# ops of one training job. For scale: GPT-5 ≈ 2e25, Mythos ≈ 1e27.
# The race's bar is normally the US's own largest run at the pause; this is
# only the floor for a catalogue with no US capacity to read.
_PC_FALLBACK_THRESHOLD = 1e28
_PC_RUN_OPTIONS = {"6-month run": "train_flop_6mo", "2-month run": "train_flop"}
_PC_HORIZON = datetime(2031, 12, 1)   # crossing-search grid end
_PC_TABLE_YEARS = (2027, 2028, 2029)  # P(crossed by EOY …) table columns
# The Pacing tab adds a third attribution: entities are countries, with China
# listed twice (mainland alone, and with Chinese labs' sites abroad).
_PC_PARTY_OPTIONS = dict(_DC_PARTY_OPTIONS, Country='country')
# Four of the milestones are dated off *released, publicly benchmarked* models
# — METR horizons, ECI scores, RLI scores. The other two are not: CoBench and
# the staff survey are internal evaluations, and Anthropic reports them for
# models it has not shipped. So when the tab's *Date points at* is anything but
# "Model release", the four have to come back by the report lag to sit on the
# same clock as everything else on the page; the other two already do.
# Sampled over the range rather than fixed, so the spread lands in the CI.
_PC_REPORT_LAG_DAYS = (30.0, 60.0)
_PC_TIMING_RELEASE = "Model release"


def _pc_report_lag(days_to, release_dated, timing_label, n=None):
    """Pull a release-dated milestone back onto the run-finished clock."""
    if not release_dated or timing_label == _PC_TIMING_RELEASE:
        return days_to
    lo, hi = _PC_REPORT_LAG_DAYS
    return days_to - np.random.uniform(lo, hi, n or len(days_to))


def _pc_ramp_for(timing_label, ramp_days):
    """The near-term discount window actually applied.

    On the release clock a crossing t days out must already have finished
    training ~a report lag earlier, so an active window extends by the lag's
    lower bound (`_PC_REPORT_LAG_DAYS[0]`). A window set to 0 stays 0 — off
    is off.
    """
    if ramp_days > 0 and timing_label == _PC_TIMING_RELEASE:
        return ramp_days + _PC_REPORT_LAG_DAYS[0]
    return ramp_days


# The tentative RSI blend. Each milestone is one operationalization of "the
# recursive-self-improvement threshold"; the weight is how much credence it
# gets, and the result is the *mixture* of their date distributions, not an
# average of their medians — a mixture keeps each component's own spread, so a
# late-but-uncertain milestone widens the answer instead of just shifting it.
# Keyed by slug because the card labels are built from the target constants.
_PC_RSI_WEIGHTS = {
    "metr_p50": 5.0,
    "metr_p80": 15.0,
    "eci_187_5": 10.0,
    "eci_200": 10.0,
    "rli_90": 15.0,
    "cobench_85": 10.0,
    "staff_10x": 8.0,
    "code_30x": 7.0,
    "nextstep_90": 10.0,
    "rev_1t": 10.0,
}
_PC_RSI_W_KEY = "pc_rsiw_"        # session-state prefix, one float per slug


_PC_RESET_KEYS = ["pc_run", "pc_pool", "pc_party",
                  "pc_timing", "pc_end_year", "pc_pause_mo", "pc_stop_dist",
                  "pc_stop_remote", "pc_withhold", "pc_dom_slow",
                  "pc_dist_when", "pc_remote_when", "pc_cn_run"] + \
                 [_PC_RSI_W_KEY + s for s in _PC_RSI_WEIGHTS]
_PC_WHEN_NOW = "Now"          # first option of the pause-scenario date sliders
_PC_CN_RUN_MAX = 12           # months: longest Chinese catch-up training run
_PC_DOM_SLOW_MAX = 90         # percent: hardest slowdown of China's own buildout
# The pause is a date the user names, in months from today — an integer, so a
# bookmarked URL can never carry a label that has gone stale (as the scenario
# cut-off sliders can); the default is the offset to a fixed *date*, so a reset
# restores that date rather than a month count.
_PC_PAUSE_MO_MAX = 60
_PC_PAUSE_DEFAULT_YM = (2027, 7)  # the plan-start date the tab opens on


def _pc_pause_default_mo(today=None):
    """Months from today to `_PC_PAUSE_DEFAULT_YM`, clamped to the slider."""
    t = today or datetime.now()
    m = ((_PC_PAUSE_DEFAULT_YM[0] - t.year) * 12
         + _PC_PAUSE_DEFAULT_YM[1] - t.month)
    return min(max(m, 0), _PC_PAUSE_MO_MAX)


_PC_DEFAULTS = {"pc_run": "2-month run",
                "pc_pool": "Nearby + announced fabric",
                "pc_party": "Tenant (who trains there)",
                "pc_timing": "Training run finished",
                "pc_end_year": _PC_HORIZON.year,
                "pc_pause_mo": _pc_pause_default_mo(),
                "pc_stop_dist": False, "pc_stop_remote": False,
                "pc_withhold": True, "pc_dom_slow": 0,
                "pc_dist_when": _PC_WHEN_NOW,
                "pc_remote_when": _PC_WHEN_NOW,
                "pc_cn_run": 2,
                **{_PC_RSI_W_KEY + s: w for s, w in _PC_RSI_WEIGHTS.items()}}
_PC_END_YEARS = [2027, 2028, 2029, 2030, 2031]


def _pc_when_options(today, end=_PC_HORIZON):
    """['Now', 'Sep 2026', …]: month labels for the cut-off date sliders.

    Month strings rather than dates so the value round-trips through the URL
    and the reset default is a constant. Built from `today`, so a stale
    bookmarked label is dropped by the caller's guard.
    """
    opts, d = [_PC_WHEN_NOW], datetime(today.year, today.month, 1)
    while True:
        d = datetime(d.year + d.month // 12, d.month % 12 + 1, 1)
        if d > end:
            return opts
        opts.append(f"{d:%b %Y}")


def _pc_cross_years(traj, grid, target):
    """Per-sample first crossing of `target` on `traj`, NaN where never.

    The same sub-step interpolation `_cc_cn_crossing_sim` does internally
    (snapping to the monthly grid swallows sub-month differences), pulled
    out so one set of sampled paths can be re-read against a second bar —
    what the Chinese-run-length comparison needs. `target` may be scalar or
    per-sample.
    """
    tgt = np.broadcast_to(np.asarray(target, dtype=float).ravel()
                          if np.ndim(target) else float(target), len(traj))
    hit = traj >= tgt[:, None]
    ok = hit.any(axis=1)
    idx = np.argmax(hit, axis=1)
    rows = np.arange(len(traj))
    prev = traj[rows, np.maximum(idx - 1, 0)]
    frac = np.clip((tgt - prev) / np.maximum(traj[rows, idx] - prev, 1e-12),
                   0.0, 1.0)
    dt = grid[1] - grid[0] if len(grid) > 1 else 0.0
    years = grid[np.maximum(idx - 1, 0)] + frac * dt
    return np.where(ok, np.where(idx == 0, 0.0, years), np.nan)


def _pc_at_years(cum, grid, years):
    """Per-sample value of a cumulative (n, len(grid)) array at `years`.

    Linear inside the step, matching `_pc_cross_years`' own interpolation, so
    a channel's contribution is read at exactly the crossing that channel
    helped produce. NaN where `years` is NaN (never crossed).
    """
    yr = np.nan_to_num(np.asarray(years, dtype=float), nan=0.0)
    dt = grid[1] - grid[0] if len(grid) > 1 else 1.0
    idx = np.clip((yr / dt).astype(int), 0, len(grid) - 2)
    rows = np.arange(len(cum))
    frac = np.clip((yr - grid[idx]) / dt, 0.0, 1.0)
    out = cum[rows, idx] + frac * (cum[rows, idx + 1] - cum[rows, idx])
    return np.where(np.isnan(years), np.nan, out)


def _pc_tri(lo, hi, n, pad=0.01):
    """Symmetric triangular draw over [lo, hi], as _cc_cn_crossing_sim does it."""
    lo, hi = min(lo, hi), max(lo, hi)
    if hi - lo < 1e-6:
        lo, hi = lo - pad, hi + pad
    return np.random.triangular(lo, 0.5 * (lo + hi), hi, n)


def _pc_add_months(d, months):
    """`d` plus whole calendar months, day clamped to the target month.

    The pause slider labels its positions with the resulting date, so the
    steps have to be exactly one month apart and never repeat a label — a
    30.44-day step cannot promise that.
    """
    m = d.month - 1 + int(months)
    y, m = d.year + m // 12, m % 12 + 1
    last = (datetime(y + m // 12, m % 12 + 1, 1) - timedelta(days=1)).day
    return d.replace(year=y, month=m, day=min(d.day, last))


def _pc_when_date(label, today):
    """A `_pc_when_options` label back to a date; 'Now' (or unknown) → today."""
    if not label or label == _PC_WHEN_NOW:
        return today
    try:
        return datetime.strptime(label, "%b %Y")
    except ValueError:
        return today


def _pc_capacity_at(rows, dcs, today, when, since=None, qs=(10, 50, 90)):
    """Each entity's largest training run at one date: label -> (q10, q50, q90).

    Runs the threshold race's own projection (`_pc_projection`) rather than a
    second fit, so the state-of-play numbers above the pause chart and the
    race table below it cannot quote different capacity for the same country,
    and reads the grid month containing `when`. Entities with nothing
    recorded are omitted.
    """
    rows = [r for r in rows if r[2]]
    if not rows or dcs is None:
        return {}
    grid, traj = _pc_projection(rows, dcs, today, since=since,
                                horizon=max(when, today + timedelta(days=31)))
    j = max(bisect.bisect_right(grid, when) - 1, 0)
    out = {}
    for label, arr in traj.items():
        col = arr[:, j]
        col = col[np.isfinite(col)]
        if len(col):
            out[label] = tuple(float(np.percentile(col, q)) for q in qs)
    return out


def _pc_entity_rows(series_shown, series_all, country_of, cluster_of,
                    unattributed=frozenset(), party='tenant'):
    """The entities racing to a threshold: [(label, kind, steps, site_names)].

    For 'tenant'/'operator', the charted companies — each one's largest
    networkable group, pooled exactly as the Data Centers tab does it, hosts
    in `unattributed` marked with †. For 'country', every country instead,
    each the largest group any one company there has, on the **unfiltered**
    site list (capacity in a country counts whoever Epoch lists in the
    building) — with China listed twice: _DC_CTY_CN_ACCESS (mainland +
    Chinese labs' sites abroad) and _DC_CTY_CN_DOMESTIC (mainland alone).
    `steps` entries are (date, value, facility).
    """
    mode = 'site' if cluster_of == {} else 'company'
    rows = []
    if party == 'country':
        entries = sorted(_dc_country_groups(series_all, country_of,
                                            'abroad').items())
        dom = [n for n in series_all if country_of.get(n) == _DC_CTY_CN]
        entries.append((_DC_CTY_CN_DOMESTIC, dom))
        for label, names in entries:
            steps = _dc_country_steps(series_all, names, mode, cluster_of)
            if steps and any(s[1] > 0 for s in steps):
                rows.append((label, 'country', steps, tuple(names)))
        return rows
    sites_of = {}
    for name, v in series_shown.items():
        for co in v.get('companies', [v['company']]):
            sites_of.setdefault(co, []).append(name)
    def _facility(names, lab):
        if lab is None:
            return names[0]
        return (f"all {len(names)} sites" if lab == "all sites"
                else f"{lab} ({len(names)} sites)")

    per_co = _dc_company_networked_series(series_shown, cluster_of)
    for co in sorted(per_co):
        s2 = [(d, v, _facility(ns, lab)) for d, v, ns, lab in per_co[co]]
        if s2 and any(s[1] > 0 for s in s2):
            label = co + " †" if co in unattributed else co
            rows.append((label, 'company', s2, tuple(sites_of.get(co, ()))))
    return rows


def _pc_plan_crossing(steps, threshold):
    """(date, facility) of the first recorded or catalogued step at or above
    `threshold` — facility is the step's detail element (the site or pooled
    group that cleared the bar), None on bare (date, value) steps. (None,
    None) when the catalogue never crosses."""
    for s in steps:
        if s[1] is not None and s[1] >= threshold:
            return s[0], (s[2] if len(s) > 2 else None)
    return None, None


def _pc_crossing_idx(traj, threshold):
    """Per-sample index of the first grid date whose value reaches `threshold`.

    `traj.shape[1]` (one past the grid) is the never-crossed sentinel; NaN
    cells never hit. Per sample the index is non-decreasing in the threshold,
    so any percentile of it is too.
    """
    hit = np.where(np.isnan(traj), -np.inf, traj) >= threshold
    any_hit = hit.any(axis=1)
    first = np.argmax(hit, axis=1)
    return np.where(any_hit, first, traj.shape[1]).astype(float)


def _pc_idx_date(idx, grid, q):
    """The q-th percentile crossing date, or None when it falls past the grid."""
    j = int(round(float(np.percentile(idx, q))))
    return grid[j] if j < len(grid) else None


def _pc_projection(rows, dcs, today, since=None, ref_steps=None,
                   horizon=None):
    """(grid, label → sampled paths) for each entity, projected exactly as the
    by-country panel does it: recorded steps, then catalogued plans under
    quality-dependent slip, then — past the ~18-month plan horizon — the US
    trend, widened by any disagreement with the entity's own fitted pace so
    the two readings show up as range rather than vanishing. The US itself
    extrapolates on its own fit.
    """
    plan_end = today + timedelta(days=_DC_CTY_PLAN_HORIZON_DAYS)
    grid = _dc_cty_month_grid(datetime(today.year, today.month, 1),
                              horizon or _PC_HORIZON)
    fits = {label: _dc_cty_fit(steps, since=since, t_end=plan_end)
            for label, _, steps, _ in rows}
    # The reference pace: the US country row when it is among the rows
    # (country attribution), else the fit of `ref_steps` — the US country
    # series the caller builds, so company entities borrow the same trend.
    us_fit = fits.get(_DC_CTY_US)
    if us_fit is None and ref_steps:
        us_fit = _dc_cty_fit(ref_steps, since=since, t_end=plan_end)
    out = {}
    for label, kind, steps, names in rows:
        own = fits[label]
        anchor = own
        if anchor is None and us_fit is not None and steps:
            # Too little history for a fit of its own: re-anchor the US pace
            # at the entity's own last step.
            t0 = min(steps[-1][0], plan_end)
            v0 = _dc_val_at([(s[0], s[1]) for s in steps], t0)
            anchor = dict(us_fit, t0=t0, v0=v0) if v0 and v0 > 0 else None
        if label == _DC_CTY_US or us_fit is None:
            pace = own
        elif own is None:
            pace = us_fit
        else:
            pace = dict(us_fit, sigma_g=max(us_fit['sigma_g'],
                                            abs(own['g'] - us_fit['g']) / 1.282))
        quality = _dc_plan_quality(dcs, names, today)
        out[label] = _dc_cty_trajectories(
            steps, anchor, grid, N_SAMPLES, pace=pace if anchor else None,
            today=today, slip_sigma=_dc_cty_slip_sigma(quality))
    return grid, out


def _pc_render_rsi_blend(components, origin, survival=None, horizon=None,
                         raw_components=None):
    """The weighted blend of the milestone ETAs, plus its own weights editor.

    `survival` (from `_pc_condition_on_today`) multiplies each entered weight
    for the mix only — with the components already truncated, that pair is
    the mixture conditioned on "nothing has crossed yet". The adjustment is
    shown two ways: the Weight column reads prior → effective share, and
    `raw_components` (the pre-conditioning draws) supply a dotted
    before-the-reality-check curve on the CDF, whose value at today is the
    mass the update removed. `horizon` (the tab's *Project through* year end)
    caps the CDF's axis. The conditioning controls render in the weights
    expander but are consumed a rerun earlier (in `_pc_render_milestones`,
    via session state), like the weights themselves.
    """
    # Read before the editor renders: Streamlit has already applied any change
    # to session state by the time this run reaches the widgets below.
    weights = {slug: float(st.session_state.get(_PC_RSI_W_KEY + slug,
                                                _PC_RSI_WEIGHTS.get(slug, 0.0)))
               for slug, _, _, _ in components}
    mix = (weights if survival is None else
           {s: w * survival.get(s, 1.0) for s, w in weights.items()})
    # One draw feeds the cards and the plot, so they cannot disagree.
    blend_days = _pc_rsi_blend_samples(components, mix, origin)
    raw_days = (None if survival is None or raw_components is None else
                _pc_rsi_blend_samples(raw_components, weights, origin))
    if blend_days is None:
        return
    early, med, late = (origin + timedelta(days=float(d))
                        for d in np.percentile(blend_days, [10, 50, 90]))

    st.subheader("RSI projection (tentative)")
    _b1, _b2 = st.columns(2)
    with _b1:
        st.metric("Blended median", med.strftime('%b %Y'))
    with _b2:
        st.metric("80% CI", f"{early:%b %Y} \u2013 {late:%b %Y}")

    _dist = _pc_rsi_dist_fig(blend_days, origin, early, med, late,
                             horizon=horizon, raw_days=raw_days)
    if _dist is not None:
        st.plotly_chart(_dist, width="stretch")

    _total = sum(weights.values()) or 1.0
    _mix_total = sum(mix.values()) or 1.0
    st.table([{
        "Milestone": lab,
        "Weight": (f"{weights[slug] / _total * 100:.0f}%" if survival is None
                   else f"{weights[slug] / _total * 100:.0f}% \u2192 "
                        f"{mix[slug] / _mix_total * 100:.0f}%"),
        "Median": _pc_eta_dates(a, d)[1].strftime('%b %Y'),
        "80% CI": "{:%b %Y} \u2013 {:%b %Y}".format(*_pc_eta_dates(a, d)[::2]),
    } for slug, lab, a, d in components])
    if survival is not None:
        _fn_caption(
            "Weight reads prior \u2192 effective credence: a Bayesian update on not "
            "seeing RSI by today. The dotted curve above is the blend before it.",
            ("a Bayesian update", "Each weight is multiplied by the share of its "
                                  "milestone's dates still possible, so a milestone "
                                  "claiming RSI should already be here loses weight "
                                  "in proportion."),
            ("The dotted curve", "Its height at today is the mass the update "
                                 "removed."))

    with st.expander("Set your own weights"):
        # Assign the defaults rather than popping the keys: a popped key is
        # re-hydrated straight back out of the URL on the next run, so the
        # reset would never take on a shared link.
        st.button("Reset weights", key="reset_pc_rsiw",
                  on_click=lambda: st.session_state.update(
                      {_PC_RSI_W_KEY + s: _PC_RSI_WEIGHTS.get(s, 0.0)
                       for s, _, _, _ in components}))
        cols = st.columns(min(len(components), 4))
        for i, (slug, lab, _, _) in enumerate(components):
            _ss_number_input(
                cols[i % len(cols)], lab, _PC_RSI_W_KEY + slug,
                _PC_RSI_WEIGHTS.get(slug, 0.0),
                min_value=0.0, max_value=100.0, step=5.0)
        st.caption(f"Entered weights total {sum(weights.values()):.0f}; they are "
                   "normalised, so any scale works. All zero falls back to the "
                   "defaults above.")
        _notyet = st.checkbox(
            "Condition milestones on “not crossed yet”",
            key="rsi_notyet",
            help="Bayesian reality check: samples dating a milestone before "
                 "today are dropped and the rest renormalized, and the blend "
                 "down-weights each milestone by the mass it put in the past. "
                 "Assumes a crossing would be known by now.")
        st.number_input(
            "Also discount the next … days", min_value=0.0, max_value=365.0,
            step=5.0, key="rsi_notyet_ramp", disabled=not _notyet,
            help="A crossing this close would already be visible in its "
                 "run-up, so “no signs today” argues against it: a sample t "
                 "days out is kept with probability t/N inside the window. "
                 "On the release clock the window extends by 30 days — a "
                 "model shipping that soon finished training ~a month "
                 "earlier. 0 keeps only the hard cut at today.")


def _pc_clock_note(release_dated, timing_label):
    """The clock sentence appended to a milestone card's hover.

    Which clock a card is on is a per-card fact (`_pc_report_lag`), so it
    belongs in that card's tooltip rather than in a shared paragraph that
    has to name every milestone on each side of the split.
    """
    if timing_label == _PC_TIMING_RELEASE:
        return " Dates are model releases."
    if not release_dated:
        return " An internal evaluation, already on this clock."
    return (" Dated off released models, so pulled back "
            f"{_PC_REPORT_LAG_DAYS[0] / 30:.0f}\u2013"
            f"{_PC_REPORT_LAG_DAYS[1] / 30:.0f} months onto the "
            f"\u201c{timing_label.lower()}\u201d clock.")


def _pc_render_milestones(timing_label, today, condition=True, ramp_days=0.0,
                          end_year=None):
    """Capabilities Milestones + the RSI blend. Rendered on the RSI tab.

    Still named `_pc_*` with the ETA helpers it calls; it moved to the RSI tab
    but the milestone machinery is unchanged. With `condition` (the sidebar's
    "not crossed yet" checkbox, default on) the cards and the blend below are
    conditioned on the present via `_pc_condition_on_today` — the cards read
    the same conditioned draws the blend mixes, so they cannot disagree.
    """
    # (slug, label, eta, release_dated, note) — see `_pc_report_lag`. The
    # note is the card's hover: what fit produced the date and what the bar
    # means, so the caption below stays one line instead of a wall of
    # caveats keyed to nothing the reader can see.
    _cap = [(f"metr_{lab}",
             f"METR {lab} horizon reaches {_PC_METR_TARGET_HRS:.0f}h",
             _pc_metr_eta(frontier_all, k, samples=True), True,
             f"The horizon at {lab[1:]}% task success. METR tab at its "
             "defaults: piecewise fit broken at GPT-4o, doubling time over "
             "[DT/2, DT\u00d72], position over the current model's own CI "
             "\u2014 the p50 slope sets the trend for both levels, so only "
             "the intercept and the position CI are re-fit here. "
             f"{_PC_METR_TARGET_HRS:.0f}h is about one work-month.")
            for lab, k in _PC_METR_LEVELS]
    _eci_fr = _eci_entity_data("US best")[1]
    # One footnote per bar, measured off the *pinned* jump ends — the same
    # pair the targets were set from, so the count a card claims and the bar
    # it dates can never disagree. Reading the far end off the live frontier
    # would make the sentence re-describe a different jump on every rescore.
    _eci_jump = {}
    _from_name, _from = _PC_ECI_JUMP_FROM
    _to_name, _to = _PC_ECI_JUMP_TO
    _jump = _to - _from
    if _jump > 0:
        for _t in _PC_ECI_TARGETS:
            _n = max(round((_t - _to) / _jump), 1)
            _eci_jump[_t] = (
                " {t:g} is {n} more jumps the size of {a} \u2192 {b} "
                "({lo:g} \u2192 {hi:g}), i.e. +{j:.1f} apiece. The jump is "
                "pinned to that pair, so a new record moves the frontier "
                "toward this bar instead of pushing it away.".format(
                    t=_t, n=_n, a=_from_name, b=_to_name,
                    lo=_from, hi=_to, j=(_t - _to) / _n))
    _cap += [(f"eci_{t:g}".replace(".", "_"), f"ECI reaches {t:g}",
              _pc_eci_eta(_eci_fr, t, samples=True), True,
              "Epoch ECI tab at its defaults: single OLS on the US-best "
              "frontier, points/yr over [PPY/2, PPY\u00d72], position "
              f"\u00b1{_PC_ECI_POS_CI:g}." + _eci_jump.get(t, ""))
             for t in _PC_ECI_TARGETS]
    _cap.append((f"rli_{_PC_RLI_TARGET_PCT:.0f}",
                 f"RLI reaches {_PC_RLI_TARGET_PCT:.0f}%",
                 _pc_rli_eta(rli_frontier_all, samples=True), True,
                 "RLI tab at its defaults: single OLS in logit space, "
                 "odds-doubling time over [DT/2, DT\u00d72] floored at 5 "
                 f"days, position \u00b1{_PC_RLI_POS_CI:g} point. "
                 f"{_PC_RLI_TARGET_PCT:.0f}% is above that tab's own "
                 "milestone table, which stops at 50%."))
    _cap.append((f"cobench_{_RSI_SUBSTITUTION_BAR:.0f}",
                 f"CoBench reaches {_RSI_SUBSTITUTION_BAR:.0f}%",
                 _pc_rsi_eta(rsi_frontier_all, samples=True), False,
                 "The CoBench fan above, at its defaults: single OLS in "
                 "logit space, odds-doubling over that fit's widened rate "
                 f"CI, position \u00b1{_PC_RSI_POS_CI:g} points. "
                 f"{_RSI_SUBSTITUTION_BAR:.0f}% is Anthropic's own stated "
                 "full-substitution bar, not a benchmark ceiling."))
    _cap.append((f"staff_{_PC_RSI_SURVEY_TARGET_X:.0f}x",
                 f"Anthropic staff acceleration \u2265{_PC_RSI_SURVEY_TARGET_X:.0f}x",
                 _pc_rsi_survey_eta(load_rsi_survey(), samples=True), False,
                 "The staff-survey fan above, at its defaults: OLS on "
                 "log(multiple) over every round fitted, the carried-over "
                 "estimated point included. The rounds do not report the "
                 f"same statistic on the same sample. "
                 f"{_PC_RSI_SURVEY_TARGET_X:.0f}x is about a doubling and a "
                 "half past the most recent round's ~4x."))
    _cap.append((f"code_{_RSI_CODE_TARGET:.0f}x",
                 f"Code per person reaches {_RSI_CODE_TARGET:.0f}x",
                 _pc_rsi_code_eta(load_rsi_code(), samples=True), False,
                 "The merged-code fan above, at its defaults: OLS on "
                 "log(multiple) over the quarters from 2025 on, doubling "
                 "time over that fit's t-widened rate CI, position over its "
                 "own residual scatter. "
                 f"{_RSI_CODE_TARGET:.0f}x is one contributor merging what "
                 f"{_RSI_CODE_TARGET:.0f} did before 2025 \u2014 lines "
                 "merged, which a coding "
                 "model inflates directly, so it is output volume rather "
                 "than research progress."))
    _cap.append((f"nextstep_{_RSI_DIR_TARGET:.0f}",
                 f"Next-step judgment reaches {_RSI_DIR_TARGET:.0f}%",
                 _pc_nextstep_eta(rsi_dir_frontier_all, samples=True), True,
                 "The research-direction fan above, at its defaults: single "
                 "OLS in logit space, odds-doubling over [DT/2, DT\u00d72] "
                 "widened to the slope's 80% t-interval, position "
                 f"\u00b1{_RSI_DIR_POS_CI:g} points (the study's own binomial "
                 f"SE on n={_RSI_DIR_N}). {_RSI_DIR_TARGET:.0f}% is the "
                 "study's own practical ceiling \u2014 what a judge shown the "
                 "finished session scores \u2014 and the sampled turns were "
                 "selected for having room for improvement."))
    _cap.append(("rev_1t", "Leading company revenue >$1T",
                 _pc_revenue_eta([_OPENAI_REVENUE, _ANTHROPIC_REVENUE],
                                 samples=True), True,
                 "Revenue tab at its defaults, per company: OLS on "
                 "log2(ARR) over every point, doubling time over "
                 "[max(10, DT\u00d70.65), DT\u00d71.5], 0.3 log2 position "
                 "\u03c3 \u2014 then whichever company crosses first. That "
                 "fit is a near-perfect 3.7-year exponential, so this is "
                 "much the tightest card here; treat the narrowness as the "
                 "fit's, not the world's."))
    # Revenue is release-dated because ARR is earned by *shipped* models, so
    # a crossing sits on the release clock exactly as a benchmark score does.
    _notes = {slug: note + _pc_clock_note(rel, timing_label)
              for slug, _l, r, rel, note in _cap if r is not None}
    _cap = [(slug, lab, r[0], _pc_report_lag(r[1], rel, timing_label))
            for slug, lab, r, rel, _n in _cap if r is not None]
    survival, _cap_raw = None, None
    ramp_days = _pc_ramp_for(timing_label, ramp_days)
    if condition:
        _cap_raw = _cap
        _cap, survival = _pc_condition_on_today(_cap, today,
                                                ramp_days=ramp_days)
    if _cap:
        st.subheader("Capabilities Milestones")
        # Two rows: all the cards on one line squeeze every label to two
        # words.
        _per_row = -(-len(_cap) // 2)
        for _start in range(0, len(_cap), _per_row):
            _chunk = _cap[_start:_start + _per_row]
            for col, (slug, lab, _anchor, _days) in zip(st.columns(_per_row),
                                                         _chunk):
                early, med, late = _pc_eta_dates(_anchor, _days)
                with col:
                    st.metric(lab, med.strftime('%b %Y'),
                              help=_notes.get(slug))
                    st.caption(f"80% CI: {early:%b %Y} \u2013 {late:%b %Y}")

        _cond_note = ("" if not condition else
                      " Dates are conditioned on the milestone not having "
                      "crossed by today.")
        st.caption(
            "Each card reproduces its own tab at that tab's defaults, so the "
            "two cannot quote different dates for the same bar \u2014 hover a "
            "card for its fit, its clock and what the bar means."
            + _cond_note)

        _pc_render_rsi_blend(_cap, today, survival,
                             horizon=(datetime(end_year, 12, 31)
                                      if end_year else None),
                             raw_components=_cap_raw)


def _pc_when(rec, horizon=None):
    """One phrase for when an entity crosses, whatever its state."""
    if rec['crossed']:
        return f"already there (since {rec['plan']:%b %Y})"
    if rec['med'] is None:
        return f"not by {(horizon or _PC_HORIZON).year} in most samples"
    return f"~{rec['med']:%b %Y}"


# METR time-horizon milestone quoted at the top of the tab: 174h, about one
# work-month (1mo = 176h in this repo's work-time units). Both reliability
# levels are cards — at a month-scale bar p50 is a credible candidate
# definition, so its earlier firing is expressed as its own card and blend
# weight rather than by exclusion (at the old 40h bar it was dropped as too
# weak, firing ~9 months ahead of everything else). `_pc_metr_eta` still fits
# on p50 (the METR tab's own default) and takes the displayed series as
# `val_key`.
_PC_METR_TARGET_HRS = 174.0
_PC_METR_LEVELS = (("p50", "p50_min"), ("p80", "p80_min"))


def _pc_eta_out(anchor_date, days_to, samples):
    """Milestone ETA output: percentile dates, or the raw samples.

    The RSI blend below mixes these distributions, so it needs the samples
    themselves — and must read the *same* draw the card above it reports,
    or the blend and its own row would disagree.
    """
    if samples:
        return anchor_date, days_to
    return _pc_eta_dates(anchor_date, days_to)


def _pc_eta_dates(anchor_date, days_to):
    return tuple(anchor_date + timedelta(days=float(d))
                 for d in np.percentile(days_to, [10, 50, 90]))


def _pc_metr_eta(frontier, val_key, target_hrs=_PC_METR_TARGET_HRS, n=None,
                 samples=False):
    """(early, median, late) dates for the METR frontier to reach `target_hrs`.

    Reproduces `render_metr()` at its defaults — piecewise linear broken at
    GPT-4o, doubling time lognormal over [DT/2, DT*2], position lognormal over
    the current model's own CI — so the two tabs can't quote different dates
    for the same milestone. As there, the p50 slope sets the trend for both
    reliability levels and only the intercept and position CI are re-fit on the
    chosen series. Returns None if the segment slope is flat or negative.
    """
    n = n or N_SAMPLES
    lo_key, hi_key = val_key[:-4] + '_lo', val_key[:-4] + '_hi'
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier[gpt4o_idx:]],
                    dtype=float)
    log2_p50 = np.array([np.log2(m['p50_min']) for m in frontier[gpt4o_idx:]])
    log2_disp = np.array([np.log2(m[val_key]) for m in frontier[gpt4o_idx:]])
    slope = fit_line(days, log2_p50)[1]
    if slope <= 0:
        return None
    dt = round(1.0 / slope)
    proj_dt = _lognormal_from_ci(max(10, round(dt / 2)), dt * 2, n)
    fitted_hrs = 2 ** (np.mean(log2_disp - slope * days) + slope * days[-1]) / 60
    cur = frontier[-1]
    pos_lo = (cur.get(lo_key) or cur[val_key]) / 60
    pos_hi = (cur.get(hi_key) or cur[val_key]) / 60
    sigma = max((np.log(pos_hi) - np.log(pos_lo)) / (2 * 1.282), 0.0)
    start_hrs = np.random.lognormal(np.log(fitted_hrs), sigma, n)
    days_to = np.log2(target_hrs / start_hrs) * proj_dt
    return _pc_eta_out(cur['date'], days_to, samples)


# The ECI companions, on the US-best frontier the ECI tab defaults to. Both
# bars are today's frontier plus **whole jumps the size of GPT-5 -> the
# current frontier** — two of them for 187.5, three for 200 — well above
# anything the ECI tab draws (its own milestone table tops out at 170) and a
# pure extrapolation of the same fit, a long way outside the frontier's
# observed range. 170 was a card once, but it sits close enough to today's
# frontier that it dated near-term model releases, not an RSI-scale
# capability — its weight moved here.
#
# `_PC_ECI_JUMP_FROM` and `_PC_ECI_JUMP_TO` are the two ends of that jump.
# **Both ends are pinned, not read off the live frontier.** The bar is
# "two more jumps the size of the one we just watched", so re-floating the
# far end onto each new record makes the bar recede as capability arrives:
# a model that beats the frontier by X raises the anchor by X and the jump
# by X, moving a two-jump bar by 3X. GPT-6 Astra did exactly that once, and
# the ECI cards moved ~13 months *later* on a refresh that made every fixed
# bar arrive earlier. Move these deliberately when the bar should mean
# something new — not on every rescore.
_PC_ECI_TARGETS = (187.5, 200.0)
_PC_ECI_JUMP_FROM = ("GPT-5", 150.0)
_PC_ECI_JUMP_TO = ("Claude Fable 5", 162.5)
_PC_ECI_POS_CI = 2.0     # the ECI tab's default position CI, fitted score +/- 2


def _pc_eci_eta(frontier, target, n=None, samples=False):
    """(early, median, late) dates for an ECI frontier to reach `target`.

    The ECI tab at its defaults — single OLS, +Pts/Yr lognormal over
    [PPY/2, PPY*2], position normal over the fitted score +/- `_PC_ECI_POS_CI`.
    Returns None if the fitted slope is flat or negative.
    """
    n = n or N_SAMPLES
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    scores = np.array([m['eci_score'] for m in frontier])
    slope = fit_line(days, scores)[1]
    if slope <= 0:
        return None
    ppy = round(slope * 365.25, 1)
    proj_dpp = _lognormal_from_ci(365.25 / round(ppy * 2, 1),
                                  365.25 / round(ppy / 2, 1), n)
    fitted = np.mean(scores - slope * days) + slope * days[-1]
    start = np.random.normal(fitted, _PC_ECI_POS_CI / 1.282, n)
    days_to = np.maximum((target - start) * proj_dpp, 0.0)
    return _pc_eta_out(frontier[-1]['date'], days_to, samples)


# The RLI companion, on the same frontier its tab charts. 90% is above the
# tab's own milestone table (which stops at 50%), so the card is a long
# extrapolation of the same logit fit.
_PC_RLI_TARGET_PCT = 90.0
_PC_RLI_POS_CI = 1.0     # the RLI tab's default position CI, fitted score +/- 1pt


def _pc_rli_eta(frontier, target_pct=_PC_RLI_TARGET_PCT, n=None, samples=False):
    """(early, median, late) dates for the RLI frontier to reach `target_pct`.

    The RLI tab at its defaults — single OLS in logit space, odds-doubling
    time lognormal over [DT/2, DT*2] (floored at 5 days as the sidebar input
    is), position normal over the fitted score +/- `_PC_RLI_POS_CI` points,
    also in logit space. Returns None if the fitted slope is flat or negative.
    """
    n = n or N_SAMPLES
    base = frontier[0]['date']
    days = np.array([(m['date'] - base).days for m in frontier], dtype=float)
    logit = _logit(np.array([m['rli_score'] / 100 for m in frontier]))
    slope = fit_line(days, logit)[1]
    if slope <= 0:
        return None
    dt = round(np.log(2) / slope)
    proj_slope = np.log(2) / _lognormal_from_ci(
        float(round(max(5.0, dt / 2), 0)), float(round(dt * 2, 0)), n)
    fitted = np.mean(logit - slope * days) + slope * days[-1]
    cur = frontier[-1]['rli_score']
    pos_lo = _logit(round(max(cur - _PC_RLI_POS_CI, 0.1), 2) / 100)
    pos_hi = _logit(round(cur + _PC_RLI_POS_CI, 2) / 100)
    start = np.random.normal(fitted, max((pos_hi - pos_lo) / (2 * 1.282), 0), n)
    days_to = np.maximum((_logit(target_pct / 100) - start) / proj_slope, 0.0)
    return _pc_eta_out(frontier[-1]['date'], days_to, samples)


# The CoBench companion, on the RSI tab's own frontier. The target is
# Anthropic's stated full-substitution bar, the milestone that tab dates.
_PC_RSI_POS_CI = 10.0    # the RSI tab's default position CI, fitted score +/- 10pts


def _pc_rsi_eta(frontier, target_pct=_RSI_SUBSTITUTION_BAR, n=None, samples=False):
    """(early, median, late) dates for the CoBench frontier to reach `target_pct`.

    The RSI tab at its defaults — single OLS in logit space, odds-doubling time
    lognormal over `_rsi_dt_ci()` (which widens the usual [DT/2, DT*2] to span
    both segment rates), position normal over the last frontier score +/-
    `_PC_RSI_POS_CI` points, also in logit space. Returns None if the fitted
    slope is flat or negative.
    """
    n = n or N_SAMPLES
    base, intercept, slope = _rsi_fit(frontier)
    if slope <= 0:
        return None
    dt_lo, dt_hi = _rsi_dt_ci(frontier, np.log(2) / slope)
    proj_slope = np.log(2) / np.maximum(_lognormal_from_ci(dt_lo, dt_hi, n), 1.0)
    cur = frontier[-1]
    fitted = intercept + slope * (cur['date'] - base).days
    pos_lo = _logit(round(max(cur['cobench'] - _PC_RSI_POS_CI, 1.0), 1) / 100)
    pos_hi = _logit(round(min(cur['cobench'] + _PC_RSI_POS_CI, 99.0), 1) / 100)
    start = np.random.normal(fitted, max((pos_hi - pos_lo) / (2 * 1.282), 0), n)
    days_to = np.maximum((_logit(target_pct / 100) - start) / proj_slope, 0.0)
    return _pc_eta_out(cur['date'], days_to, samples)


# The staff-survey companion: when do Anthropic researchers self-report this
# much speedup. 10x is roughly a doubling and a half past the most recent
# round's ~4x, so it sits just outside the measured range rather than deep in
# the extrapolation the RSI tab's own caption warns about.
_PC_RSI_SURVEY_TARGET_X = 10.0


def _pc_rsi_survey_eta(rows, target_x=_PC_RSI_SURVEY_TARGET_X, n=None,
                       samples=False):
    """(early, median, late) dates for self-reported speedup to reach `target_x`.

    The RSI tab's survey fan at its defaults — OLS on log(multiple) over
    every round the tab fits (the carried-over `estimated` point included),
    doubling time lognormal over `_rsi_survey_dt_ci()`'s t-widened interval,
    position lognormal over the fitted multiple divided and multiplied by
    `_RSI_SURVEY_POS_FACTOR`. Returns None if the fitted slope is flat or
    negative.
    """
    n = n or N_SAMPLES
    base = rows[0]['date']
    days = np.array([(r['date'] - base).days for r in rows], dtype=float)
    logs = np.log(np.array([r['uplift'] for r in rows]))
    intercept, slope = fit_line(days, logs)
    if slope <= 0:
        return None
    dt = np.log(2) / slope
    proj_slope = np.log(2) / np.maximum(
        _lognormal_from_ci(*_rsi_survey_dt_ci(rows, dt), n=n), 1.0)
    fitted = intercept + slope * (rows[-1]['date'] - base).days
    start = np.random.normal(fitted, np.log(_RSI_SURVEY_POS_FACTOR) / 1.282, n)
    days_to = np.maximum((np.log(target_x) - start) / proj_slope, 0.0)
    return _pc_eta_out(rows[-1]['date'], days_to, samples)


# The merged-code companion: when does output per Anthropic engineer reach
# `_RSI_CODE_TARGET`x the pre-2025 baseline. Reuses the section's own fit and
# rate CI, so the card and the fan cannot date it differently.


def _pc_rsi_code_eta(rows, target_x=_RSI_CODE_TARGET, n=None, samples=False):
    """(early, median, late) dates for merged code per person to reach `target_x`.

    The section's fan at its defaults — OLS on log(multiple) over the quarters
    from 2025 on, doubling time lognormal over `_rsi_code_dt_ci()`, position
    lognormal over the fitted multiple divided and multiplied by
    `_RSI_CODE_POS_FACTOR`. Returns None if the fitted slope is flat.
    """
    n = n or N_SAMPLES
    base, icpt, slope = _rsi_code_fit(rows)
    if slope <= 0:
        return None
    proj_slope = np.log(2) / np.maximum(
        _lognormal_from_ci(*_rsi_code_dt_ci(rows, np.log(2) / slope), n=n), 1.0)
    cur = rows[-1]
    fitted = icpt + slope * (cur['date'] - base).days
    start = np.random.normal(fitted, np.log(_RSI_CODE_POS_FACTOR) / 1.282, n)
    days_to = np.maximum((np.log(target_x) - start) / proj_slope, 0.0)
    return _pc_eta_out(cur['date'], days_to, samples)


# The research-direction companion: when does the model's next step beat the
# researcher's in four turns of five. Reuses the RSI tab's own fit and rate CI
# rather than fitting its own, so the two cannot quote different dates.


def _pc_nextstep_eta(frontier, target_pct=_RSI_DIR_TARGET, n=None,
                     samples=False):
    """(early, median, late) dates for the detour study to reach `target_pct`.

    The section's fan at its defaults — single OLS in logit space,
    odds-doubling time lognormal over `_rsi_dir_dt_ci()`, position normal over
    the last frontier score +/- `_RSI_DIR_POS_CI` points. Returns None if the
    fitted slope is flat or negative.
    """
    n = n or N_SAMPLES
    base, intercept, slope = _rsi_dir_fit(frontier)
    if slope <= 0:
        return None
    dt_lo, dt_hi = _rsi_dir_dt_ci(frontier, round(np.log(2) / slope))
    proj_slope = np.log(2) / np.maximum(_lognormal_from_ci(dt_lo, dt_hi, n), 1.0)
    cur = frontier[-1]
    fitted = intercept + slope * (cur['date'] - base).days
    sigma = (_logit((cur['better'] + _RSI_DIR_POS_CI) / 100)
             - _logit((cur['better'] - _RSI_DIR_POS_CI) / 100)) / (2 * 1.282)
    start = np.random.normal(fitted, max(float(sigma), 0), n)
    days_to = np.maximum((_logit(target_pct / 100) - start) / proj_slope, 0.0)
    return _pc_eta_out(cur['date'], days_to, samples)


# The revenue companion, and the only milestone here that isn't a benchmark:
# ARR is a revealed-preference bar. $1T is the Revenue tab's own top
# milestone (`_REV_MILESTONES`) — roughly a third of today's US software
# sector bought from one seller, which is hard to explain unless the product
# substitutes for labour rather than assisting it.
_PC_REV_TARGET_B = 1000.0


def _pc_revenue_eta(series, target_b=_PC_REV_TARGET_B, n=None, samples=False):
    """(early, median, late) dates for the *leading* company's ARR to reach
    `target_b` ($B).

    Reproduces `render_revenue()` at its defaults for each company — OLS on
    log2(ARR) over every point, doubling time lognormal over
    [max(10, DT x 0.65), DT x 1.5], position normal on the fitted last value
    with the tab's 0.3 log2 sigma — and then takes whichever company gets
    there first **per sample**, which is what "the leading company" means:
    the milestone is a claim about the frontier of commercial scale, not
    about a particular seller.

    The two companies' draws are independent. They plainly aren't in
    reality, but a common shock would narrow the answer on nothing measured,
    and independence is the assumption that doesn't quietly do that.

    `series` is a list of raw (date, ARR) tables. Returns None when no
    company has a positive fitted slope. The dates come out on the release
    clock — ARR is earned by shipped models — so the caller lags them like
    a benchmark score.
    """
    n = n or N_SAMPLES
    anchor = max(_parse_revenue(d)[0][-1] for d in series)
    best = None
    for data in series:
        dates, vals = _parse_revenue(data)
        if len(vals) < 3:
            continue
        base = dates[0]
        days = np.array([(d - base).days for d in dates], dtype=float)
        intercept, slope = fit_line(days, np.log2(np.array(vals)))
        if slope <= 0:
            continue
        dt = 1.0 / slope
        proj_dt = _lognormal_from_ci(max(10.0, dt * 0.65), dt * 1.5, n)
        start = np.random.normal(intercept + slope * days[-1], 0.3, n)
        days_to = np.maximum(np.log2(target_b) - start, 0.0) * proj_dt
        # Onto one anchor, so the per-sample minimum compares like with like:
        # the companies' series end on different dates.
        days_to += (dates[-1] - anchor).days
        best = days_to if best is None else np.minimum(best, days_to)
    if best is None:
        return None
    return _pc_eta_out(anchor, best, samples)


def _pc_condition_on_today(components, today, ramp_days=0.0):
    """Condition each milestone on not having crossed by `today`.

    Every ETA sampler already truncates at its own anchor (`days_to >= 0`);
    this moves the truncation point to the present, by rejection rather than
    clamping: samples dating the crossing at or before today are dropped and
    the survivors are the conditional distribution P(T = t | T > today).

    `ramp_days` extends the update into the near future as a soft likelihood
    rather than a wider hard cut: a sample `t` days out is kept with
    probability t/ramp_days inside the window (1 beyond it) — the closer a
    crossing, the more visible its run-up would already be, so "no signs
    today" argues against it in proportion. 0 disables the ramp.

    Returns (conditioned components, {slug: survival fraction}); a component
    with no surviving samples drops out entirely. The blend must then mix
    with each weight multiplied by its survival — survival is the expected
    likelihood, so that pair is exactly the mixture conditioned on the
    observation, and a definition that put mass in the ruled-out region
    loses credence in proportion.
    """
    out, survival = [], {}
    for slug, lab, anchor, days in components:
        rel = days + (anchor - today).days
        if ramp_days > 0:
            keep = np.random.random(len(days)) < np.clip(
                rel / float(ramp_days), 0.0, 1.0)
        else:
            keep = rel > 0
        survival[slug] = float(np.mean(keep))
        if keep.any():
            out.append((slug, lab, anchor, days[keep]))
    return out, survival


def _pc_rsi_blend_samples(components, weights, origin, n=None):
    """Mix the milestone ETAs into one distribution, in days from `origin`.

    `components` is [(slug, label, anchor_date, days_to)]; weights need not sum
    to anything in particular — zero-weight components drop out, and all-zero
    falls back to the default weighting rather than dividing by zero. Returns
    None if nothing is left to mix.
    """
    n = n or N_SAMPLES
    picks = [(c, max(float(weights.get(c[0], 0.0)), 0.0)) for c in components]
    total = sum(w for _, w in picks)
    if total <= 0:
        picks = [(c, _PC_RSI_WEIGHTS.get(c[0], 0.0)) for c in components]
        total = sum(w for _, w in picks)
    picks = [(c, w) for c, w in picks if w > 0]
    if not picks:
        return None
    # Everything on one clock so components with different anchors can mix.
    pools = [np.array([(c[2] - origin).days], dtype=float) + c[3]
             for c, _ in picks]
    p = np.array([w for _, w in picks], dtype=float) / total
    idx = np.random.choice(len(pools), size=n, p=p)
    out = np.empty(n, dtype=float)
    for i, pool in enumerate(pools):
        m = idx == i
        if m.any():
            out[m] = np.random.choice(pool, int(m.sum()), replace=True)
    return out


def _pc_rsi_blend(components, weights, origin, n=None):
    """(early, median, late) dates for the blend. See `_pc_rsi_blend_samples`."""
    out = _pc_rsi_blend_samples(components, weights, origin, n)
    if out is None:
        return None
    return tuple(origin + timedelta(days=float(d))
                 for d in np.percentile(out, [10, 50, 90]))


def _pc_rsi_dist_fig(days, origin, early, med, late, horizon=None,
                     raw_days=None):
    """Cumulative probability that the blend has landed by each date.

    A CDF rather than a histogram: the blend is a mixture of milestones,
    so a density is lumpy where its components sit and the bin width becomes a
    presentation choice. The cumulative curve has neither problem, and reading
    "X% by date D" off it is the question this section answers. Sampled daily
    so the hover tracks continuously. `horizon` (the tab's *Project through*
    year end) sets where the axis stops; without one it falls back to the
    98th percentile, past which the curve is a flat tail that only costs
    width. A median past the horizon simply isn't annotated — the curve
    ending below 50% already says so. `raw_days` (the unconditioned blend)
    draws as a dotted ghost behind the main curve; the grid starts at the
    earlier of the two 0.5th percentiles so the ghost's nonzero value at
    today — the mass the reality check removed — is visible.
    """
    srt = np.sort(days)
    d0 = float(np.percentile(srt, 0.5))
    d1 = (float((horizon - origin).days) if horizon is not None
          else float(np.percentile(srt, 98)))
    raw_srt = np.sort(raw_days) if raw_days is not None else None
    if raw_srt is not None:
        d0 = min(d0, float(np.percentile(raw_srt, 0.5)))
    if d1 - d0 < 2:
        return None
    grid = np.arange(np.floor(d0), np.ceil(d1) + 1, 1.0)
    cdf = np.searchsorted(srt, grid, side='right') / len(srt) * 100
    dates = [origin + timedelta(days=float(g)) for g in grid]

    fig = go.Figure()
    if raw_srt is not None:
        raw_cdf = np.searchsorted(raw_srt, grid, side='right') \
            / len(raw_srt) * 100
        fig.add_trace(go.Scatter(
            x=dates, y=raw_cdf.tolist(), mode='lines',
            line=dict(color='#9aa5b1', width=1.5, dash='dot'),
            hovertext=[f"By {d:%b %d, %Y}<br>{v:.0f}% before reality check"
                       for d, v in zip(dates, raw_cdf)],
            hoverinfo='text', showlegend=False))
    fig.add_trace(go.Scatter(
        x=dates, y=cdf.tolist(), mode='lines',
        line=dict(color='#4F8DFD', width=2.5),
        fill='tozeroy', fillcolor='rgba(79,141,253,0.12)',
        hovertext=[f"By {d:%b %d, %Y}<br>{v:.0f}% chance"
                   for d, v in zip(dates, cdf)],
        hoverinfo='text', showlegend=False))
    fig.add_vrect(x0=early, x1=late, fillcolor='rgba(52,152,219,0.10)',
                  line_width=0, layer='below')
    # add_vline's own annotation averages the x endpoints, which throws on a
    # datetime axis; place it separately against the paper y-axis instead.
    if horizon is None or med <= horizon:
        fig.add_vline(x=med, line=dict(color='#2c3e50', width=2, dash='dash'))
        fig.add_annotation(x=med, y=1.0, yref='paper', yanchor='bottom',
                           xanchor='left', xshift=4, showarrow=False,
                           text=f"median {med:%b %Y}",
                           font=dict(size=11, color='#2c3e50'))
    fig.update_layout(
        height=300, margin=dict(l=50, r=30, t=30, b=35),
        font=dict(color='#1a1a2e'),
        xaxis=dict(range=[dates[0], dates[-1]],
                   gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   showspikes=True, spikemode='across', spikethickness=1,
                   spikedash='dot', spikecolor='#9aa5b1',
                   tickfont=dict(color='#1a1a2e')),
        yaxis=dict(title="Chance crossed by then", range=[0, 100],
                   ticksuffix='%', gridcolor='rgba(0,0,0,0.1)', zeroline=False,
                   tickfont=dict(color='#1a1a2e'),
                   title_font=dict(color='#1a1a2e')),
        hovermode='x', plot_bgcolor='white', paper_bgcolor='white')
    return fig


# Realized ship lag of a US frontier model — run finished to public release,
# prep plus queue: GPT-5.5 Pro's run finished ~Feb 2026 and shipped Apr 23
# (Mythos model card). The released frontier trails the trained one by this,
# so a bar meaning "best model *trained* by the pause" sits this much climb
# above the released-frontier extrapolation. Deliberately larger than the
# 30-day `_CC_RUN_COMPLETION_LAG` prep constant, which excludes the queue.
_PC_SHIP_LAG_DAYS = 60

# The ECI->METR bridge is a fit over today's models; a pause dated years out
# maps to horizons of centuries, which is noise wearing a number's clothes.
# Past this the state-of-play table says ">10y" instead.
_PC_METR_CAP_HRS = 10 * 2000


# Display names for the four rate terms, in _CC_CHANNELS order.
_PC_CHANNEL_LABELS = {
    'compute': "Compute — China's own cluster buildout",
    'compute_domestic': "Compute — domestic clusters",
    'compute_abroad': "Compute — remote access to clusters abroad",
    'innovation': "Indigenous innovation — never dries up",
    'diffusion': "Diffusion — published US methods, drying up after "
                 "pacing plan",
    'distillation': "Distillation — training against US model outputs",
}


def _pc_fmt_horizon(hours):
    """A METR horizon for the state-of-play table, capped at
    `_PC_METR_CAP_HRS` — the ECI->METR fit is not worth reading in
    centuries. The cap reads "10y+", not ">10y": st.table renders markdown,
    and a leading ">" turns the cell into a blockquote."""
    return (fmt_hrs(hours) if hours <= _PC_METR_CAP_HRS
            else f"{fmt_hrs(_PC_METR_CAP_HRS)}+")


def _pc_render_why(chan, grid0, traj0, target0, years, years_base, *,
                   cn_gain, cn_extra, cn_run_mo, us_run_mo, anchor_name,
                   anchor_d, anchor_eci, horizon_year, notes=()):
    """Bottom-of-panel breakdown: which channel closed how much of the gap.

    Two readings of the same sampled paths, because neither alone answers
    "why X months". *ECI closed* is each rate term's cumulative contribution
    at its own sample's crossing (they sum to the gap exactly — that's the
    accounting identity the sim is built on). *Without it* re-crosses the
    path with that term's cumulative subtracted, which is the number a
    policy question actually asks, and is deliberately **not** additive: the
    channels interact, so removing compute leaves the gap wider and the
    distillation term stronger for longer. Both are stated as such.

    `years` is the final crossing (run-length shift included), `years_base`
    the same without that shift, `target0`/`traj0`/`grid0` the organic bar
    and paths the channels accumulated against.
    """
    ok = np.isfinite(years)
    if ok.sum() < 100:
        return
    total_mo = float(np.median(years[ok])) * 12.0

    def _delta(alt):
        """Median paired months between an alternative crossing and the real
        one; None when the alternative mostly never crosses."""
        both = np.isfinite(alt) & ok
        return (float(np.median((alt - years)[both])) * 12.0
                if both.sum() >= 100 else None)

    # Compute splits in two when the sim carried a domestic-only shadow:
    # what domestic clusters alone would have contributed, and the rest —
    # the export-controlled channel. The two still sum to the compute term,
    # so the identity below is untouched.
    cum = {k: chan[k] for k in _CC_CHANNELS}
    if 'compute_domestic' in chan:
        cum['compute_domestic'] = chan['compute_domestic']
        cum['compute_abroad'] = chan['compute'] - chan['compute_domestic']
        del cum['compute']
    contrib, rows = {}, []
    for k, arr in cum.items():
        contrib[k] = float(np.nanmedian(
            _pc_at_years(arr, grid0, years - cn_extra)))
    if cn_run_mo != us_run_mo:
        contrib['run'] = cn_gain
    total_eci = sum(contrib.values())
    for k, eci in sorted(contrib.items(), key=lambda kv: -kv[1]):
        if k == 'run':
            label = (f"Longer training run — {cn_run_mo} mo instead of "
                     f"{us_run_mo}")
            d = _delta(years_base)
        else:
            label = _PC_CHANNEL_LABELS[k]
            d = _delta(_pc_cross_years(traj0 - cum[k], grid0, target0)
                       + cn_extra)
        rows.append({
            "Channel": label,
            "ECI closed": f"{eci:+.1f}",
            "Share": f"{eci / total_eci:.0%}" if total_eci else "—",
            "Without it": (f"not by {horizon_year}" if d is None else
                           f"{d:+.1f} mo"),
        })
    rows.append({"Channel": f"**Total** — the whole gap, closed over "
                            f"**{total_mo:.1f} months** from "
                            f"{pretty(anchor_name)} ({anchor_d:%b %Y}, ECI "
                            f"{anchor_eci:.0f})",
                 "ECI closed": f"{total_eci:+.1f}",
                 "Share": "100%", "Without it": "—"})
    st.markdown("##### Why that long?")
    st.table(rows)
    _fn_caption(
        f"The {total_mo:.1f} months run from China's last frontier model, not "
        "from the pacing plan start. <b>ECI closed</b> and "
        "<b>Without it</b> answer different questions."
        + ("".join(" " + n for n in notes) if notes else ""),
        ("ECI closed", "Each term's cumulative contribution at its own sample's "
                       "crossing. The terms sum to the gap by construction, "
                       "which is why the shares are exact. Same samples as the "
                       "chart, so this is a decomposition rather than a second "
                       "model."),
        ("Without it", "Re-crosses the same paths with that term removed "
                       "\u2014 the number a policy question asks. It does not "
                       "add up: kill compute and the gap stays wider, so "
                       "distillation keeps running at full strength longer and "
                       "the shortfalls partly cover for each other. Median over "
                       "samples, so a row's two columns need not agree in "
                       "rank."))


def _pc_render_us_pause(today, pause_d, caps, run_days=_DAYS_2MO,
                        us_steps=None, timing_label=_DC_TIMING_OPTIONS[0],
                        horizon=None):
    """If the US paused: China's catch-up to the paused frontier, in ECI.

    The pause is a **date the user names** (`pc_pause_mo`, rendered by
    `render_pacing` so the bar it implies is available to both halves of
    the tab), not a compute threshold: the question the panel answers is
    "if the US stopped *then*, how long until China is level", and the
    compute scale reached by then is an output of that choice rather than
    its input. That scale is what the *Compute Thresholds* race below then
    uses as its bar — there is no separate threshold control.

    `pause_d` is the pause on the run-finished clock and `caps` the
    `_pc_capacity_at()` reading of each country at that date, both from the
    caller. `us_steps` is the sidebar-pooled US country series
    (capacity-online dates, run-length units) so the climb pace follows the
    tab's networking selector; single-site fallback without it.

    Until the pause the US frontier climbs at the compute-derived rate, and
    the bar China must cross is the best model the US has *trained* by then:
    that climb plus `_PC_SHIP_LAG_DAYS` of extra climb, since the trained
    frontier leads the released one (that the climb is calibrated on) by the
    realized ship lag. The paused stock stays distillable while a gap
    remains, then China runs on its indigenous algorithmic rate
    (_cc_innovation_algo_band) plus its compute term alone.

    The pause arrives on the **run-finished** clock, the panel's natural
    one; the date shown carries `timing_label`'s offset like every other
    date here, which is what keeps the US-China gap milestone-invariant
    (see below).

    Dates follow `timing_label` (the sidebar's *Date points at*): both
    countries' events shift in lockstep to the chosen milestone — run
    starts / run finishes / model releases — so the US–China gap never
    depends on the setting. Internally the math runs on natural clocks (the
    pause at run completion, China's ECI data at release dates); the
    display offsets are `shift − run` for compute-side events and
    `shift − run − 30d` for release-dated ones. No stochastic release-queue
    wait is added anywhere (deliberately unlike the CC tab's 161 section):
    a queue delays both countries alike, so it would widen the window
    without moving the gap.

    `horizon` is the sidebar's *Project through* date: it bounds the sim's
    search grid and the chart's x-axis alike, so this panel ends where the
    timeline chart above it does. A crossing past it is reported as "not
    by <year>" rather than searched for anyway.
    """
    # Lockstep milestone offsets (see docstring).
    shift = _dc_timing_shift(timing_label, run_days)
    off_us = timedelta(days=shift - run_days)
    off_cn = timedelta(days=shift - run_days - _CC_RUN_COMPLETION_LAG.days)
    d_pause_v = pause_d + off_us
    # Streamlit renders a container where it was created, so the
    # state-of-play table lands here — between the date the caller drew and
    # the question the rest of the panel answers.
    state_box = st.container()
    st.subheader("If the US paced: when does China catch up?")
    st.markdown("##### Policy levers")
    cb1, cb2, cb3 = st.columns(3)
    withhold = cb1.checkbox(
        "US does not give access to strongest models at plan start",
        key="pc_withhold",
        value=True,
        help="A release freeze: once the final threshold run starts, the US "
             "ships nothing new, so the best queryable teacher is the last "
             "generation released before run start — one run's climb (plus "
             "the ship lag) below the bar. Uncheck to let the US keep "
             "serving right up to (and including) the paused frontier.")
    _lvl = _cc_cn_level_offset(load_eci_compute(_mtime=_eci_mtime()))
    stop_dist = cb2.checkbox(
        "Stop Chinese distillation as of today", key="pc_stop_dist",
        help="API-level controls that block training against US model "
             "outputs: the distillation channel is cut from today onward, "
             "instead of decaying only as the gap closes. The cut stops the "
             "forward rate only — measured distillation is mostly a level "
             + (f"(~{_lvl[0]:+.0f} ECI at matched compute and date vs US "
                "models, iso-compute rates indistinguishable), "
                if _lvl else "advantage, ")
             + "and points already banked stay banked.")
    stop_remote = cb3.checkbox(
        "Cut Chinese remote access to compute abroad", key="pc_stop_remote",
        help="Chinese labs lose the DayOne Johor-class sites and rented "
             "clusters abroad: their largest run falls back to the biggest "
             "domestic cluster (a level setback the domestic buildout must "
             "first regrow), and grows at the domestic catalogued pace "
             "thereafter.")
    dom_slow_pct = st.slider(
        "Slow China's domestic compute growth by", 0, _PC_DOM_SLOW_MAX,
        value=_PC_DEFAULTS["pc_dom_slow"], step=1, format="%d%%",
        key="pc_dom_slow",
        help="Equipment, HBM and fab controls that bite on the buildout "
             "itself: China's own clusters grow this much slower than the "
             "catalogued domestic pace. The cut comes off the domestic "
             "share of the compute term, so it bites whether or not remote "
             "access is cut \u2014 and a slower domestic pace also makes the "
             "remote-access setback take longer to regrow. 0% = the "
             "catalogued buildout.")
    with st.expander("Advanced"):
        st.markdown("**When the controls bite**")
        _opts = _pc_when_options(today)
        for _k in ("pc_dist_when", "pc_remote_when"):
            if st.session_state.get(_k, _PC_WHEN_NOW) not in _opts:
                st.session_state.pop(_k, None)
        s1, s2 = st.columns(2)
        dist_when = s1.select_slider(
            "Distillation cut off", options=_opts,
            value=_PC_DEFAULTS["pc_dist_when"], key="pc_dist_when",
            disabled=not stop_dist,
            help="Enforcement takes time. Until this date China keeps "
                 "distilling under the gap-decay law; after it the channel "
                 "is gone. Needs the checkbox above.")
        remote_when = s2.select_slider(
            "Remote access cut off", options=_opts,
            value=_PC_DEFAULTS["pc_remote_when"], key="pc_remote_when",
            disabled=not stop_remote,
            help="China's compute grows on the export-control band until "
                 "this date, then takes the setback and continues at the "
                 "domestic pace — so a later cut costs more OOM to regrow. "
                 "Needs the checkbox above.")
        st.caption("Both default to *Now*, which is what the checkboxes on "
                   "their own mean. Moving a slider right buys China more "
                   "time on that channel before it closes.")
        st.divider()
        st.markdown("**Length of Chinese training run**")
        us_run_mo = max(int(round(run_days / 30)), 1)
        if not (us_run_mo <= st.session_state.get("pc_cn_run", us_run_mo)
                <= _PC_CN_RUN_MAX):
            st.session_state.pop("pc_cn_run", None)   # stale for this bar
        cn_run_mo = st.slider(
            "Months China trains its catch-up model for", us_run_mo,
            _PC_CN_RUN_MAX, value=us_run_mo, step=1, key="pc_cn_run",
            help=f"The bar is a {us_run_mo}-month US run, but China need "
                 "not match its length. Running longer puts more total "
                 "compute into one model and needs proportionally less "
                 "cluster to do it — so the hardware is there sooner — "
                 "while the run itself finishes later. It can go either "
                 "way; the cards above show which.")
        _fn_caption(
            f"Defaults to {us_run_mo} \u2014 matching the bar, and no change from "
            "the panel above. A longer run trades compute against wall clock.",
            ("A longer run trades compute against wall clock",
             f"An L-month run carries \u00d7L/{us_run_mo} the compute of the "
             f"{us_run_mo}-month bar: China's whole path lifts by what that buys "
             f"and shifts right by L\u2212{us_run_mo} months."))
    d_dist = _pc_when_date(dist_when, today)
    d_remote = _pc_when_date(remote_when, today)
    cc_rows = load_eci_compute(_mtime=_eci_mtime())
    eci_all = load_eci_frontier(_mtime=_eci_mtime())
    us_fr = _cc_country_frontier(eci_all, 'United States of America')
    cn_fr = _cc_country_frontier(eci_all, 'China')
    dec = _cc_decomp(cc_rows)
    inno = _cc_innovation_algo_band(cc_rows)
    if len(us_fr) < 2 or len(cn_fr) < 2 or dec is None or inno is None:
        st.info("Not enough ECI data for the pause scenario.")
        return
    us_best = max(us_fr, key=lambda x: x[1])
    cn_best = max(cn_fr, key=lambda x: x[1])
    anchor_d, anchor_eci, anchor_name = cn_best
    if anchor_eci >= us_best[1]:
        st.success(f"{pretty(anchor_name)} is already at the US frontier.")
        return

    # Frontier-grade coefficient pair, as in _cc_us_vs_china (pooled fallback).
    a_partial, b_algo = _cc_pooled_decomp(cc_rows)
    fgm = _cc_frontier_grade_algo(
        cc_rows, load_eci_frontier(_mtime=_eci_mtime(), full_window=True))
    if fgm:
        a_partial, b_algo = fgm['a_partial'], fgm['b_time']
    # A longer Chinese run is a one-off *level* move, not a faster rate: it
    # puts ×(L/L_us) the compute into one model, worth a_partial per ×10 of
    # capability, and needs proportionally less cluster so the hardware is
    # there sooner — but that model lands L−L_us months later. So China's
    # whole deliverable path lifts by `cn_gain` and shifts right by
    # `cn_extra`. Which way the crossing moves is the point of the control:
    # the lift is worth cn_gain / (China's ECI rate) of time, and beyond a
    # few months the extra wall clock outruns it. The organic sim is
    # untouched — the algorithmic channels don't speed up because a run is
    # longer — so the lift is applied by reading the same sampled paths
    # against a bar lowered by cn_gain.
    cn_gain = a_partial * float(np.log10(cn_run_mo / us_run_mo))
    cn_extra = (cn_run_mo - us_run_mo) * 30.44 / 365.25
    us_algo, _, _ = _cc_iso_compute_rate(cc_rows, 'United States of America')
    cn_algo, _, _ = _cc_iso_compute_rate(cc_rows, 'China')
    if us_algo is None or cn_algo is None:
        a_lo = a_mid = a_hi = b_algo
    else:
        a_lo, a_hi = min(us_algo, cn_algo), max(us_algo, cn_algo)
        a_mid = cn_algo
    # The US pace and hardware clock come from the sidebar-pooled US series
    # (capacity-online dates, run-length units) — the same roster the table
    # above races — falling back to the single-site country fit. σ-sampled,
    # so the pause date and bar carry the pace uncertainty.
    plan_end = today + timedelta(days=_DC_CTY_PLAN_HORIZON_DAYS)
    chk_pace = _cc_country_pace_check(today)
    us_fit, pooled = None, False
    if us_steps:
        us_fit = _dc_cty_fit(us_steps, since=_DC_DEFAULTS["dc_cty_since"],
                             t_end=plan_end)
        pooled = us_fit is not None
    if us_fit is None:
        us_fit = chk_pace.get(_DC_CTY_US)
    n_s = N_SAMPLES
    if us_fit is not None:
        g_s = np.maximum(
            np.random.normal(us_fit['g'], us_fit['sigma_g'], n_s), 0.05)
        us_rate = b_algo + a_partial * us_fit['g']
        us_rate_s = np.maximum(b_algo + a_partial * g_s, 1.0)
    else:
        g_s = None
        us_rate = _cc_frontier_eci_slope(us_fr, datetime(2024, 1, 1)) or 15.0
        us_rate_s = np.full(n_s, us_rate)
    g_mid = 0.5 * (_CC_CN_COMPUTE_LO + _CC_CN_COMPUTE_HI)
    pace_lo, pace_hi, _obs = _cc_cn_pace_band(cn_fr, a_mid + a_partial * g_mid)
    # The pause is the date the slider names, so every sample pauses
    # together and only the climb to it is uncertain — the frozen bar
    # carries the pace spread, the date does not.
    t_today = max((today - anchor_d).days, 0) / 365.25
    t_pause_s = np.full(n_s, max((pause_d - anchor_d).days, 0) / 365.25)
    t_pause_s = np.maximum(t_pause_s, t_today)
    # The bar is the best model the US has *trained* by the pause — the
    # 1e28 flagship itself. The released frontier trails the trained one by
    # the realized ship lag (_PC_SHIP_LAG_DAYS), and the climb is measured
    # from us_best's own release date (the time grid is anchored on China's
    # last release, ~a month later — omitting that offset understated the
    # released level too). Per sample, so pace uncertainty widens both the
    # date and the bar.
    _us_dt = ((anchor_d - us_best[0]).days + _PC_SHIP_LAG_DAYS) / 365.25
    level_s = us_best[1] + us_rate_s * (t_pause_s + _us_dt)
    lvl50 = float(np.median(level_s))
    # Best *queryable* model under the release freeze: the released frontier
    # as of the pause run's start — the freeze means nothing new ships while
    # the final run proceeds, so distillation dries at this level, about one
    # run's climb (plus the ship lag) below the bar. A released level, so no
    # ship-lag term — only the anchor-date offset.
    teacher_s = us_best[1] + us_rate_s * (
        np.maximum(t_pause_s - run_days / 365.25, 0.0)
        + (anchor_d - us_best[0]).days / 365.25)
    teach50 = float(np.median(teacher_s))
    # Three-channel algorithmic engine: distillation decays with the gap,
    # diffusion over ~a year after the pause (published stock absorbed),
    # innovation never. The checkboxes tighten two channels: distillation
    # cut at `d_dist` (not at the gap's close), and remote compute cut at
    # `d_remote` as a *level setback* — losing the sites abroad drops
    # China's largest run back to its biggest domestic cluster, so the
    # compute term contributes nothing until the domestic buildout regrows
    # the lost OOMs at its own catalogued pace, and runs at that pace
    # thereafter. A cut dated later costs *more* to regrow (the two paces
    # diverge until then) but leaves the band untouched until it bites, so
    # a later cut is never worse for China. The pace band above stays on
    # the default band on purpose — it is shared with the CC tab's
    # crossing, and a scenario toggle must not re-tune the reality-check
    # factor.
    g_lo_eff, g_hi_eff = _CC_CN_COMPUTE_LO, _CC_CN_COMPUTE_HI
    dom_keep = 1.0 - dom_slow_pct / 100.0
    comp_dead, comp_slow, dlvl_oom = None, None, 0.0
    g_dom_lo = g_dom_hi = None       # domestic band, for the Assumes line
    comp_shadow = None               # domestic-only compute, for the breakdown
    t_cut_r = max((d_remote - anchor_d).days, 0) / 365.25
    chk_dom = chk_pace.get(_DC_CTY_CN_DOMESTIC)
    if chk_dom is not None:
        # The domestic-only counterfactual is computed whether or not the
        # checkbox is on: with it off it is the shadow the breakdown
        # subtracts to price China's access to compute abroad; with it on
        # it also sets the run's own band and setback.
        _ser = _dc_series_for_metric(dc_all, 'train_flop')
        _cty = {dc['name']: _dc_site_country(dc) for dc in dc_all}
        _grp = _dc_country_groups(_ser, _cty, 'abroad')

        def _cur_level(names):
            vals = [v for d, v, *_ in
                    _dc_country_steps(_ser, names, 'site', {})
                    if d <= today and v > 0]
            return vals[-1] if vals else None

        acc = _cur_level(_grp.get(_DC_CTY_CN_ACCESS, []))
        dom = _cur_level([n for n in _ser if _cty.get(n) == _DC_CTY_CN])
        g_dom = max(chk_dom['g'], 0.05)
        g_dom_hi = min(g_hi_eff, g_dom)
        g_dom_lo = min(g_lo_eff, g_dom_hi)
        if dom_keep < 1.0:
            # The domestic-growth lever. Compute growth is a rate in OOM/yr
            # and the run's band is the domestic buildout plus whatever
            # access abroad adds on top, so the cut comes off the *domestic
            # component* of both bands: the export-control band keeps the
            # abroad excess, while a run already confined to domestic
            # clusters takes the cut in full. Scaling the whole band
            # instead would charge the slowdown twice over for capacity
            # the policy doesn't touch.
            g_hi_eff = max(g_hi_eff - g_dom_hi * (1.0 - dom_keep), 0.0)
            g_lo_eff = max(g_lo_eff - g_dom_lo * (1.0 - dom_keep), 0.0)
            g_dom_hi *= dom_keep
            g_dom_lo *= dom_keep
            # Floored so the regrowth window below stays finite.
            g_dom = max(g_dom * dom_keep, 0.01)
        dlvl_today = (max(float(np.log10(acc / dom)), 0.0)
                      if acc and dom and acc > dom else 0.0)
        comp_shadow = (a_partial * g_dom_hi,
                       (t_today, t_today + dlvl_today / g_dom)
                       if dlvl_today else None)
        if stop_remote:
            if t_cut_r <= t_today:
                # Bites now: the whole sim runs on the domestic band.
                g_lo_eff, g_hi_eff = g_dom_lo, g_dom_hi
            else:
                # Export-control band until the cut, domestic band after.
                comp_slow = (t_cut_r,
                             a_partial * _pc_tri(g_dom_lo, g_dom_hi, n_s))
            if dlvl_today:
                # The setback is the gap at the cut date: today's gap,
                # widened by how far the two paces diverge before it bites.
                dlvl_oom = max(dlvl_today + (g_mid - g_dom)
                               * max(t_cut_r - t_today, 0.0), 0.0)
                comp_dead = (max(t_cut_r, t_today),
                             max(t_cut_r, t_today) + dlvl_oom / g_dom)
    elif dom_keep < 1.0:
        # No catalogued domestic pace to decompose the band against, so the
        # whole band takes the cut — the conservative reading of the lever.
        g_lo_eff, g_hi_eff = g_lo_eff * dom_keep, g_hi_eff * dom_keep
    pure = _cc_pure_innovation_band(cc_rows)
    kw = dict(us_anchor=us_best[1], us_rate=us_rate_s, us_pause_level=level_s,
              a_partial=a_partial, g_lo=g_lo_eff,
              g_hi=g_hi_eff, algo_lo=a_lo, algo_mid=a_mid,
              algo_hi=a_hi, pace_lo=pace_lo, pace_hi=pace_hi,
              n=n_s, t_pause=t_pause_s, comp_dead=comp_dead,
              comp_slow=comp_slow,
              t_dist_stop=(max((d_dist - anchor_d).days, 0) / 365.25
                           if stop_dist else None),
              dist_teacher=teacher_s if withhold else None,
              **({'pure_lo': pure[0], 'pure_hi': pure[1]} if pure else {}))
    if horizon is not None:
        # Search only as far as the sidebar's projection range, keeping at
        # least a year of grid so a near-term horizon still draws.
        kw['horizon_yrs'] = max((horizon - anchor_d).days / 365.25, 1.0)
    chan = {}
    years, grid_yrs, traj = _cc_cn_crossing_sim(
        anchor_eci, level_s, inno_lo=inno[0], inno_hi=inno[1],
        channels=chan, comp_shadow=comp_shadow, **kw)
    # The run-length trade-off, read off the *same* sampled paths so the
    # comparison is paired (no MC noise between the two answers): the bar
    # drops by cn_gain, then the extra wall clock is added back. The organic
    # paths are kept for the breakdown, whose channels accumulated on them.
    years_base, traj0, grid0, target0 = years, traj, grid_yrs, level_s
    if cn_run_mo != us_run_mo:
        target0 = level_s - cn_gain
        years = _pc_cross_years(traj, grid_yrs, target0) + cn_extra
        traj = traj + cn_gain
        grid_yrs = grid_yrs + cn_extra
    # ── State of play at plan start: what each side has when the music
    # stops, written into the slot under the slider. Compute comes from the
    # threshold race's own projection (so the two sections cannot disagree),
    # capability from the same sampled ECI paths the chart draws, and the
    # METR horizon from the ECI->METR bridge the ECI tabs use. ──
    cn_pause_s = _pc_at_years(traj0, grid0, t_pause_s)

    def _state_row(label, eci_s, behind=None):
        e10, e50, e90 = (float(np.nanpercentile(eci_s, q))
                         for q in (10, 50, 90))
        c = caps.get(label)
        return {
            "At plan start": label,
            "Largest training run": (
                f"{_log_op(c[1])} log OP ({_log_op(c[0])}–{_log_op(c[2])})"
                if c else "—"),
            "Frontier ECI": f"~{e50:.0f} ({e10:.0f}–{e90:.0f})",
            "METR horizon (p50)": _pc_fmt_horizon(
                _eci_to_metr_p50_min(e50)[0] / 60),
            "METR horizon (p80)": _pc_fmt_horizon(
                _eci_to_metr_p80_min(e50)[0] / 60),
            "Behind the US": ("—" if behind is None
                              else f"~{behind:.0f} mo" if np.isfinite(behind)
                              else "—"),
        }

    def _us_date_at(score):
        """When the US line on the chart below reaches `score`: the recorded
        frontier where it covers the score, this panel's own climb beyond
        it. Read off the same line the chart draws, so "months behind" and
        the picture agree."""
        if score <= us_best[1]:
            return _ecg_frontier_date_at_score(
                [(d, sc) for d, sc, _n in us_fr], score)
        return us_best[0] + timedelta(days=(score - us_best[1])
                                      / max(us_rate, 1e-6) * 365.25)

    def _behind_us(eci_s):
        _at = _us_date_at(float(np.nanmedian(eci_s)))
        return ((pause_d - _at).days / 30.44 if _at is not None
                else float('nan'))

    state_rows = [_state_row(_DC_CTY_US, level_s),
                  _state_row(_DC_CTY_CN_ACCESS, cn_pause_s,
                             _behind_us(cn_pause_s))]
    # Domestic-only China is the same sampled paths with the compute the
    # sites abroad bought subtracted — the shadow term the breakdown below
    # prices — so the two Chinese rows can never disagree with each other.
    if 'compute_domestic' in chan:
        dom_pause_s = _pc_at_years(
            traj0 - (chan['compute'] - chan['compute_domestic']),
            grid0, t_pause_s)
        state_rows.append(_state_row(_DC_CTY_CN_DOMESTIC, dom_pause_s,
                                     _behind_us(dom_pause_s)))
    with state_box:
        st.table(state_rows)

    # Sensitivity: China's compute term at the catalogued China-accessible
    # buildout pace (Chinese labs' sites abroad included) instead of the
    # export-control band — the channel where "remote" compute would bite.
    # Moot when the remote-access checkbox has cut that channel.
    chk_ca = chk_pace.get(_DC_CTY_CN_ACCESS)
    d50_ca = None
    if chk_ca and not stop_remote:
        y_ca, _gc, _tc = _cc_cn_crossing_sim(
            anchor_eci, level_s, inno_lo=inno[0], inno_hi=inno[1],
            **dict(kw, g_lo=max(chk_ca['g'] - chk_ca['sigma_g'], 0.05),
                   g_hi=chk_ca['g'] + chk_ca['sigma_g']))
        if cn_run_mo != us_run_mo:
            y_ca = _pc_cross_years(_tc, _gc, level_s - cn_gain) + cn_extra
        ok_ca = y_ca[np.isfinite(y_ca)]
        if len(ok_ca) >= 100:
            d50_ca = anchor_d + timedelta(
                days=float(np.percentile(ok_ca, 50)) * 365.25)
    yr_ok = years[np.isfinite(years)]
    if len(yr_ok) < 100:
        st.info(f"China does not reach the paced US frontier by "
                f"{horizon.year} in most samples — widen *Project through*."
                if horizon is not None else
                "Sampled rates were too weak to give a crossing date.")
        return
    d10, d50, d90 = (anchor_d + timedelta(days=float(np.percentile(yr_ok, p))
                                          * 365.25) for p in (10, 50, 90))
    # Milestone offsets came from the top of the function (the slider's
    # caption needs them); the pause is naturally a run-finished date and
    # China's ECI data are release dates, so both display at the sidebar's
    # milestone and their gap never depends on the setting.
    v10, v50, v90 = d10 + off_cn, d50 + off_cn, d90 + off_cn
    # Months from the pause to the crossing on a common milestone, matched
    # per sample so both uncertainties propagate. The release-prep lag nets
    # out of the gap (rung-invariant: both offsets move together).
    _floor = np.maximum(t_pause_s * 365.25, (today - anchor_d).days)
    _delta_mo = ((years * 365.25 - _floor)[np.isfinite(years)]
                 - _CC_RUN_COMPLETION_LAG.days) / 30.44
    s10, s50, s90 = (float(np.percentile(_delta_mo, p)) for p in (10, 50, 90))
    m1, m2 = st.columns(2)
    m1.metric(f"China reaches paced US frontier "
              f"(plan start {d_pause_v:%b %Y}, ECI ~{lvl50:.0f})",
              f"{v50:%b %Y}", f"{v10:%b %Y} – {v90:%b %Y} (80%)",
              delta_color="off")
    m2.metric("Time for China to surpass",
              f"~{s50:.1f} mo", f"{s10:.1f} – {s90:.1f} mo (80%)",
              delta_color="off")
    _mo = run_days // 30
    _clock = {
        "Data center construction":
            f"**Dates = training runs start** (cluster online); the {_mo}-mo "
            f"run finishes {_mo} mo later, a release ~1 mo after that.",
        "Training run finished":
            "**Dates = training runs finish**; a release follows ~1 mo "
            "later.",
        "Model release":
            f"**Dates = model releases** ({_mo}-mo run + ~1 mo prep).",
    }[timing_label]
    _fn_caption(
        _clock + " Both countries move together, so the gap never changes.",
        ("Both countries move together", "Pick the milestone with the sidebar's "
                                         "*Date points at*. No release-queue "
                                         "wait is added anywhere, since a queue "
                                         "delays both sides alike."))
    if cn_run_mo != us_run_mo:
        _both = np.isfinite(years) & np.isfinite(years_base)
        _net = float(np.median((years - years_base)[_both])) * 12 \
            if _both.any() else 0.0
        _xtra = cn_run_mo - us_run_mo
        _mos = f"{_xtra} month" + ("s" if _xtra != 1 else "")
        _fn_caption(
            f"<b>{cn_run_mo}-month Chinese run</b> (<i>Advanced</i>): "
            f"\u00d7{cn_run_mo / us_run_mo:.1f} the compute in one model against "
            f"{_mos} of extra training, for a net "
            f"<b>{abs(_net):.1f} mo {'later' if _net > 0 else 'earlier'}</b> than "
            f"matching the bar's {us_run_mo} months.",
            ("the compute in one model", f"Worth +{cn_gain:.1f} ECI, or "
                                         f"{_xtra - _net:.1f} months of China's "
                                         "climb."),
            (f"{_mos} of extra training", "On the chart China's fan therefore starts "
                                          "that much after its last actual model, "
                                          "and that much higher."))
    _when_r = "today" if d_remote <= today else f"**{d_remote:%b %Y}**"
    _dom_hi = g_hi_eff if g_dom_hi is None else g_dom_hi
    _dom_lo = g_lo_eff if g_dom_hi is None else g_dom_lo
    _slow = (f"; domestic buildout slowed {dom_slow_pct}%"
             if dom_slow_pct else "")
    if comp_dead is not None:
        _asm_comp = (f"compute — from {_when_r}, falls back "
                     f"**{dlvl_oom:.1f} OOM** to the largest domestic "
                     f"cluster, ~{comp_dead[1] - comp_dead[0]:.1f} yr to "
                     f"regrow at ×{10 ** _dom_hi:.1f}/yr "
                     f"(checkbox{_slow})")
    elif stop_remote:
        _asm_comp = (f"compute — domestic pace only from {_when_r}, "
                     f"×{10 ** _dom_lo:.1f}–{10 ** _dom_hi:.1f}/yr "
                     f"(checkbox{_slow})")
    else:
        _asm_comp = (f"compute — keeps growing ×{10 ** g_lo_eff:.1f}–"
                     f"{10 ** g_hi_eff:.1f}/yr (export-control band{_slow})")
    if stop_dist:
        _asm_dist = ("distillation — **cut today** (checkbox)"
                     if d_dist <= today else
                     f"distillation — **cut {d_dist:%b %Y}** (checkbox); "
                     "the gap-decay law until then")
    elif withhold:
        _asm_dist = ("distillation — release freeze from pause-run start "
                     f"(checkbox): the teacher is the last pre-freeze "
                     f"release (ECI ~{teach50:.0f}), so it dries up well "
                     "below the bar")
    else:
        _asm_dist = ("distillation — the paused frontier itself stays "
                     "queryable, fading only as China closes the gap")
    def _strip(txt):
        """The channel strings lead with their own name; the marker is that
        name, so drop the prefix rather than saying it twice."""
        body = txt.split("\u2014", 1)[-1].strip().replace("**", "")
        return body[0].upper() + body[1:] + "."

    _fn_line(
        "**Assumes** secure weights, a pause on methods as well as models, and "
        "these settings for China's distillation and compute:",
        ("secure weights", "China can query the paused US models but never "
                           "steal them \u2014 theft would be instant "
                           "catch-up."),
        ("a pause on methods", "The pause stops new US models and methods "
                               "alike: published know-how is fully absorbed "
                               f"~{_CC_DIFF_ABSORB_YRS:.0f} yr after the "
                               "pause."),
        ("distillation", _strip(_asm_dist)),
        ("compute", _strip(_asm_comp)))

    # ── The race, in ECI: US climbs to the bar and freezes; China's fan
    # catches up and crosses. The whole chart rides the sidebar milestone:
    # every date (actual points included) carries its side's lockstep
    # offset, so the diamond, the window and the metric cards agree. ──
    # The x-axis ends at the sidebar's projection range (as on every other
    # tab), falling back to just past the 90th-percentile crossing.
    x_end = horizon if horizon is not None else v90 + timedelta(days=150)
    grid_d = [anchor_d + off_cn + timedelta(days=y * 365.25)
              for y in grid_yrs]
    keep = [i for i, d in enumerate(grid_d) if d <= x_end]
    _hover = (lambda d, s, n: f"{pretty(n)}<br>ECI {s:.0f}"
              + (f"<br>released {d:%b %Y}" if off_cn.days else ""))
    fig = go.Figure()
    _dc_add_projection_band(fig, today + off_cn, x_end)
    fig.add_vrect(x0=v10, x1=v90, fillcolor='rgba(214,39,40,0.10)',
                  line_width=0, layer='below')
    fig.add_annotation(x=v50, y=1.0, yref='paper', yanchor='bottom',
                       text='80% crossing window', showarrow=False,
                       font=dict(size=10, color='#7F1010'))
    # US: actual frontier, then the climb to the bar, flat once it paces.
    fig.add_trace(go.Scatter(
        x=[d + off_cn for d, s, n in us_fr], y=[s for d, s, n in us_fr],
        mode='lines+markers', line=dict(color='#1F77B4', width=1.5),
        marker=dict(size=4, color='#1F77B4', line=dict(color='white', width=0.5)),
        text=[_hover(d, s, n) for d, s, n in us_fr],
        hoverinfo='text', name='US (actual)'))
    # Two visually distinct segments: the projected climb (dashed, like every
    # projection) and the frozen bar the plan holds (thick translucent level).
    t0_us = max(us_best[0], anchor_d) + off_cn
    if d_pause_v > t0_us:
        fig.add_trace(go.Scatter(
            x=[t0_us, d_pause_v], y=[us_best[1], lvl50], mode='lines',
            line=dict(color='#1F77B4', width=2, dash='dash'),
            name='US climb (projected)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(
        x=[max(d_pause_v, t0_us), x_end], y=[lvl50, lvl50], mode='lines',
        line=dict(color='#1F77B4', width=4.5), opacity=0.4,
        name=f'US paced at {lvl50:.0f}', hoverinfo='skip'))
    if d_pause_v > anchor_d + off_cn:
        fig.add_annotation(x=d_pause_v, y=lvl50,
                           text=f'US paces ~{d_pause_v:%b %Y}',
                           showarrow=False, yshift=12,
                           font=dict(size=10, color='#1F77B4'))
    # China: actual frontier, then the sim's fan.
    fig.add_trace(go.Scatter(
        x=[d + off_cn for d, s, n in cn_fr], y=[s for d, s, n in cn_fr],
        mode='lines+markers', line=dict(color='#D62728', width=1.5),
        marker=dict(size=4, color='#D62728', line=dict(color='white', width=0.5)),
        text=[_hover(d, s, n) for d, s, n in cn_fr],
        hoverinfo='text', name='China (actual)'))
    pct = {p: np.percentile(traj, p, axis=0) for p in (10, 25, 50, 75, 90)}
    xs = [grid_d[i] for i in keep]
    for lo_p, hi_p, alpha in ((10, 90, 0.16), (25, 75, 0.20)):
        fig.add_trace(go.Scatter(
            x=xs + xs[::-1], mode='lines',
            y=[pct[hi_p][i] for i in keep] + [pct[lo_p][i] for i in keep][::-1],
            fill='toself', fillcolor=f'rgba(214,39,40,{alpha})',
            line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=[pct[50][i] for i in keep], mode='lines',
        line=dict(color='#7F1010', width=2.5), name='China (projected median)',
        hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra>China median</extra>'))
    fig.add_trace(go.Scatter(
        x=[v50], y=[lvl50], mode='markers',
        marker=dict(symbol='diamond', size=13, color='#D62728',
                    line=dict(color='white', width=1.5)),
        name=f'median crossing {v50:%b %Y}', hoverinfo='name'))
    _xtitle = {"Data center construction": "Training runs start",
               "Training run finished": "Training runs finish",
               "Model release": "Model releases"}[timing_label]
    fig.update_layout(
        height=440, plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=55, r=20, t=20, b=40), font=dict(color='#222222'),
        legend=dict(font=dict(size=11, color='#222'), x=0.01, y=0.99,
                    bgcolor='rgba(255,255,255,0.75)', bordercolor='#DDD',
                    borderwidth=1),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)', title_text=_xtitle,
                   range=[datetime(2024, 1, 1) + off_cn, x_end],
                   tickfont=dict(color='#222'), title_font=dict(color='#222')),
        yaxis=dict(title_text="Frontier ECI score", gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222'), title_font=dict(color='#222')))
    st.plotly_chart(fig, use_container_width=True)

    _rate_src = ((f"({b_algo:.0f} algo + {a_partial:.0f} pts/×10 × the "
                  f"{'sidebar-pooled' if pooled else 'largest-site'} US "
                  f"series' ×{10 ** us_fit['g']:.1f}/yr pace, σ-sampled)")
                 if us_fit else "(observed US frontier slope)")
    _cf_notes = [
        ("the US climbs", f"~{us_rate:.0f} ECI/yr {_rate_src}. Pausing later "
                          "freezes a better model, so the bar rises with the "
                          "slider even though China gets the same head "
                          "start."),
        ("China must cross", "Once distillation and diffusion dry up China "
                  "runs on "
                  "indigenous innovation "
                  + (f"({pure[0]:.0f}\u2013{pure[1]:.0f} ECI/yr)" if pure else
                     f"(floor {inno[0]:.0f}\u2013{inno[1]:.0f} ECI/yr)")
                  + " plus its compute term. Same sim and pace band as the "
                  "Compute/capabilities/diffusion crossing section, which "
                  f"lands earlier: its bar is lower ({_CC_CN_TARGET_ECI:.0f}), "
                  "its clock starts at China's last release rather than at "
                  "the pause, distillation never dries up there, and it adds "
                  "a release-queue wait this panel omits."),
    ]
    _cf_tail = ""
    if d50_ca is not None:
        _cf_tail = ", and a faster Chinese buildout would move that"
        _cf_notes.append(
            ("a faster Chinese buildout",
             "If Chinese labs sustained the catalogued China-accessible "
             f"buildout pace (\u00d7{10 ** chk_ca['g']:.1f}/yr, sites abroad "
             "included \u2014 a from-zero ramp) instead of the "
             "export-control band, the crossing moves to "
             f"~{d50_ca + off_cn:%b %Y}."))
    _fn_caption(
        "Counterfactual: the US climbs until the date the slider names, "
        f"freezing at <b>ECI ~{lvl50:.0f}</b> (median) \u2014 the bar China "
        "must cross" + _cf_tail + ".", *_cf_notes)

    # ── Why that long? The same samples, decomposed by rate term. ──
    _notes = []
    if comp_shadow is not None:
        _notes.append(
            "The two compute rows split China's cluster growth into what "
            "its domestic buildout alone would have delivered and what "
            "access to clusters abroad adds on top — the same "
            "domestic-only counterfactual the remote-access checkbox "
            "applies, which is why ticking it zeroes that row.")
    if stop_dist:
        _notes.append(
            "Distillation is cut "
            + ("today" if d_dist <= today else f"{d_dist:%b %Y}")
            + ", so its row counts only what it closed before then.")
    if comp_dead is not None:
        _notes.append(
            f"Compute contributes nothing between "
            f"{anchor_d + timedelta(days=comp_dead[0] * 365.25):%b %Y} and "
            f"{anchor_d + timedelta(days=comp_dead[1] * 365.25):%b %Y} "
            f"while the {dlvl_oom:.1f} OOM setback is regrown.")
    elif stop_remote:
        _notes.append("Compute runs at the domestic pace only.")
    if dom_slow_pct:
        _notes.append(
            f"China's own clusters grow {dom_slow_pct}% slower than the "
            "catalogued domestic pace, which lowers both compute rows.")
    _notes.append(f"Every row is scaled by the pace band "
                  f"(×{pace_lo:.2f}–{pace_hi:.2f}), the reality check "
                  "against China's own observed frontier slope.")
    _pc_render_why(
        chan, grid0, traj0, target0, years, years_base,
        cn_gain=cn_gain, cn_extra=cn_extra, cn_run_mo=cn_run_mo,
        us_run_mo=us_run_mo, anchor_name=anchor_name, anchor_d=anchor_d,
        anchor_eci=anchor_eci,
        horizon_year=(horizon.year if horizon is not None else _PC_HORIZON.year),
        notes=_notes)


def render_pacing():
    _today = datetime.now()

    # ── Sidebar ──
    with st.sidebar:
        st.header("Pacing")
        # A bookmarked URL can carry a stale label; drop it rather than raise.
        if st.session_state.get("pc_run") not in _PC_RUN_OPTIONS:
            st.session_state.pop("pc_run", None)
        run_label = st.radio(
            "Run length", list(_PC_RUN_OPTIONS),
            index=list(_PC_RUN_OPTIONS).index(_PC_DEFAULTS["pc_run"]),
            key="pc_run",
            help="How long the job occupies the cluster; a longer run clears "
                 "the bar on less hardware.")
        if st.session_state.get("pc_timing") not in _DC_TIMING_OPTIONS:
            st.session_state.pop("pc_timing", None)
        timing_label = st.selectbox(
            "Date points at", list(_DC_TIMING_OPTIONS),
            index=list(_DC_TIMING_OPTIONS).index(_PC_DEFAULTS["pc_timing"]),
            key="pc_timing",
            help="Same milestones as the Data Centers tab: capacity online, "
                 "+ one run (the chosen length) for training finished, "
                 "+ 30 days more for a release.")
        if st.session_state.get("pc_pool") not in _DC_NETWORK_OPTIONS:
            st.session_state.pop("pc_pool", None)
        net_label = st.selectbox(
            "Data centers networked together", list(_DC_NETWORK_OPTIONS),
            index=list(_DC_NETWORK_OPTIONS).index(_PC_DEFAULTS["pc_pool"]),
            key="pc_pool",
            help="Same levels as the Data Centers tab: how many sites one "
                 "entity can drive as a single training job.")
        if st.session_state.get("pc_party") not in _PC_PARTY_OPTIONS:
            st.session_state.pop("pc_party", None)
        party_label = st.radio(
            "Attribute each site to", list(_PC_PARTY_OPTIONS),
            index=list(_PC_PARTY_OPTIONS).index(_PC_DEFAULTS["pc_party"]),
            key="pc_party",
            help="Tenant credits a site to every user Epoch lists — "
                 "Colossus 2 counts for Anthropic, Cursor and SpaceXAI "
                 "alike — falling back to the owner; operator credits the "
                 "owner alone (Colossus → SpaceXAI, Stargate → Oracle). "
                 "Country races countries instead, with China shown both "
                 "as the mainland alone and as China-accessible (mainland "
                 "+ Chinese labs' sites abroad).")
        with st.expander("Projection range"):
            pc_end_year = st.radio(
                "Project through", _PC_END_YEARS, horizontal=True,
                index=_PC_END_YEARS.index(_PC_DEFAULTS["pc_end_year"]),
                key="pc_end_year")
        if st.button("Reset", key="pc_reset"):
            for k in _PC_RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state.update(_PC_DEFAULTS)
            st.rerun()

    key = _PC_RUN_OPTIONS[run_label]
    basis = _DC_NETWORK_OPTIONS[net_label]
    party = _PC_PARTY_OPTIONS[party_label]
    pc_horizon = datetime(pc_end_year, 12, 1)

    st.header("Pacing")
    st.warning("Warning: under construction, not final.")

    # ── Series shared by both halves of the tab ──
    run_days = _DAYS_6MO if key == 'train_flop_6mo' else _DAYS_2MO
    shift_days = _dc_timing_shift(timing_label, run_days)
    dc_view = _dc_with_party(dc_all, party)
    series_unshifted = _dc_series_for_metric(dc_view, key, cap_date=None)
    cluster_of = ({} if basis == 'none' else None if basis == 'all'
                  else _dc_network_site_clusters(basis))
    country_of = {dc['name']: _dc_site_country(dc) for dc in dc_view}
    _mode = 'site' if cluster_of == {} else 'company'
    _groups = _dc_country_groups(series_unshifted, country_of, 'abroad')
    _us_names = _groups.get(_DC_CTY_US, [])
    _cn_names = _groups.get(_DC_CTY_CN_ACCESS, [])
    _dom_names = [n for n in series_unshifted
                  if country_of.get(n) == _DC_CTY_CN]
    _us_steps_raw = _dc_country_steps(series_unshifted, _us_names, _mode,
                                      cluster_of)
    _cn_steps_raw = _dc_country_steps(series_unshifted, _cn_names, _mode,
                                      cluster_of)
    _dom_steps_raw = _dc_country_steps(series_unshifted, _dom_names, _mode,
                                       cluster_of)

    # ── The pause date drives the whole tab, so it is drawn here rather
    # than inside the panel: the capacity the US has reached by then is
    # also the bar the threshold race below runs to. Months from today
    # under the hood — an int round-trips through the URL and can never go
    # stale — with every position labelled by the date it means. ──
    _months = list(range(_PC_PAUSE_MO_MAX + 1))
    if st.session_state.get("pc_pause_mo") not in _months:
        st.session_state.pop("pc_pause_mo", None)
    pause_mo = st.select_slider(
        "US starts pacing plan", options=_months,
        value=_PC_DEFAULTS["pc_pause_mo"], key="pc_pause_mo",
        format_func=lambda m: f"{_pc_add_months(_today, m):%b %Y}",
        help="The US stops training new frontier models on this date and "
             "never restarts. Everything below follows from it — the "
             "compute and capability each side has when the music stops, "
             "how long China then needs, and the bar the *Compute "
             "Thresholds* race runs to.")
    pause_d = _pc_add_months(_today, pause_mo)
    caps = _pc_capacity_at(
        [(_DC_CTY_US, 'country', _us_steps_raw, tuple(_us_names)),
         (_DC_CTY_CN_ACCESS, 'country', _cn_steps_raw, tuple(_cn_names)),
         (_DC_CTY_CN_DOMESTIC, 'country', _dom_steps_raw,
          tuple(_dom_names))],
        dc_view, _today, pause_d, since=_DC_DEFAULTS["dc_cty_since"])
    # The race's bar is the run the US is mounting when it pauses, so the
    # two halves of the tab cannot describe different scales. Same units as
    # the racers (the sidebar's run length), so run length moves both.
    threshold = (caps[_DC_CTY_US][1] if _DC_CTY_US in caps
                 else _PC_FALLBACK_THRESHOLD)
    threshold_label = f"{_log_op(threshold)} log OP"

    _pc_render_us_pause(_today, pause_d, caps, run_days,
                        us_steps=_us_steps_raw, timing_label=timing_label,
                        horizon=pc_horizon)

    # ── Entities and projections ──
    series_all = series_unshifted
    if shift_days:
        series_all = {n: {**v, 'pts': [(d + timedelta(days=shift_days), val)
                                       for d, val in v['pts']]}
                      for n, v in series_all.items()}
    hidden = _dc_hidden_companies(dc_view, now=_today)
    series_shown = {n: v for n, v in series_all.items()
                    if v['company'] not in hidden}
    rows = _pc_entity_rows(series_shown, series_all, country_of, cluster_of,
                           unattributed=_dc_unattributed_companies(dc_view),
                           party=party)
    ref_steps = _dc_country_steps(series_all, _us_names, _mode, cluster_of)
    grid, traj = _pc_projection(rows, dc_view, _today,
                                since=_DC_DEFAULTS["dc_cty_since"],
                                ref_steps=ref_steps, horizon=pc_horizon)

    recs = []
    for label, kind, steps, names in rows:
        plan_d, plan_via = _pc_plan_crossing(steps, threshold)
        idx = _pc_crossing_idx(traj[label], threshold)
        recs.append({
            'label': label, 'kind': kind, 'plan': plan_d, 'via': plan_via,
            'crossed': plan_d is not None and plan_d <= _today,
            'med': _pc_idx_date(idx, grid, 50),
            'lo': _pc_idx_date(idx, grid, 10),
            'hi': _pc_idx_date(idx, grid, 90),
            'share': float((idx < len(grid)).mean()),
            'p_by': {yr: float((idx <= bisect.bisect_right(
                         grid, datetime(yr, 12, 31)) - 1).mean())
                     for yr in _PC_TABLE_YEARS},
        })
    recs.sort(key=lambda r: (
        (0, r['plan']) if r['crossed'] else
        (1, r['med'] or pc_horizon + timedelta(days=1)), -r['share']))

    # ── Headline ──
    st.subheader("Compute Thresholds")
    us = next((r for r in recs if r['label'] == _DC_CTY_US), None)
    cn = next((r for r in recs if r['label'] == _DC_CTY_CN_ACCESS), None)
    if us is not None and cn is not None:
        head = (f"**United States {_pc_when(us, pc_horizon)}; "
                f"China-accessible {_pc_when(cn, pc_horizon)}")
        d_us = us['plan'] if us['crossed'] else us['med']
        d_cn = cn['plan'] if cn['crossed'] else cn['med']
        if d_us is not None and d_cn is not None:
            gap = (d_cn - d_us).days / 30.44
            head += (f" \u2014 {abs(gap):.0f} months "
                     f"{'later' if gap >= 0 else 'earlier'}.**")
        else:
            head += ".**"
        st.markdown(head)

    # ── Timeline chart ──
    fig = go.Figure()
    order = [r['label'] for r in recs]
    for i, r in enumerate(recs):
        if r['kind'] == 'country':
            color = _DC_CTY_COLORS.get(
                r['label'], _DC_CTY_OTHER_COLORS[i % len(_DC_CTY_OTHER_COLORS)])
        else:
            color = _dc_color(r['label'], i)
        y = [r['label']]
        if r['crossed']:
            fig.add_trace(go.Scatter(
                x=[r['plan']], y=y, mode='markers',
                marker=dict(color=color, size=10, symbol='circle'),
                showlegend=False, hoverinfo='text',
                hovertext=[f"<b>{r['label']}</b><br>crossed "
                           f"{r['plan']:%b %Y}"
                           + (f" with {r['via']}" if r['via'] else "")]))
            continue
        lo, hi = r['lo'], r['hi']
        if lo is not None:
            fig.add_trace(go.Scatter(
                x=[lo, hi or grid[-1]], y=y * 2, mode='lines',
                line=dict(color=color, width=5), opacity=0.35,
                showlegend=False, hoverinfo='skip'))
        if r['med'] is not None:
            rng = (f"{lo:%b %Y} – " + (f"{hi:%b %Y}" if hi else
                                       f">{pc_horizon.year}")) if lo else "—"
            fig.add_trace(go.Scatter(
                x=[r['med']], y=y, mode='markers',
                marker=dict(color=color, size=11, symbol='diamond'),
                showlegend=False, hoverinfo='text',
                hovertext=[f"<b>{r['label']}</b><br>median {r['med']:%b %Y} · "
                           f"80%: {rng}<br>P(cross by {pc_horizon.year}) = "
                           f"{r['share']:.0%}"]))
        else:
            fig.add_trace(go.Scatter(
                x=[grid[-1]], y=y, mode='markers',
                marker=dict(color=color, size=10, symbol='triangle-right-open'),
                showlegend=False, hoverinfo='text',
                hovertext=[f"<b>{r['label']}</b><br>median beyond "
                           f"{pc_horizon.year} · P(cross by "
                           f"{pc_horizon.year}) = {r['share']:.0%}"]))
        if r['plan'] is not None:
            fig.add_trace(go.Scatter(
                x=[r['plan']], y=y, mode='markers',
                marker=dict(color=color, size=9, symbol='circle-open'),
                showlegend=False, hoverinfo='text',
                hovertext=[f"<b>{r['label']}</b><br>plan first "
                           f"clears the bar {r['plan']:%b %Y}"
                           + (f" with {r['via']}" if r['via'] else "")]))
    fig.add_shape(type='line', x0=_today, x1=_today, yref='paper', y0=0, y1=1,
                  line=dict(color='gray', dash='dot', width=1))
    fig.add_annotation(x=_today, yref='paper', y=1.04, text="today",
                       showarrow=False, font=dict(size=11, color='gray'))
    x_lo = min([_today] + [r['plan'] for r in recs if r['crossed']])
    # Ends at the sidebar's projection range, like every other tab's chart —
    # so both graphs here span the same window and a "not crossed" marker
    # sits at the year the P(cross) column is quoted for.
    x_hi = grid[-1] + timedelta(days=120)
    fig.update_yaxes(categoryorder='array',
                     categoryarray=list(reversed(order)))
    fig.update_xaxes(range=[x_lo - timedelta(days=90), x_hi])
    fig.update_layout(height=max(340, 60 + 30 * len(recs)),
                      margin=dict(l=10, r=10, t=30, b=10),
                      title=dict(text=f"First ≥{threshold_label} run, by "
                                      "entity", font=dict(size=15)))
    st.plotly_chart(fig, use_container_width=True)
    _fn_caption(
        "Filled dot = already crossed \u00b7 open dot = the plan's crossing "
        "\u00b7 diamond + band = projected median and 80% range"
        + (" \u00b7 each country is its largest networkable group"
           if party == 'country' else " \u00b7 \u2020 = no recorded tenant"),
        ("diamond + band", "Plan slip plus a fitted trend past ~18 months "
                           "\u2014 the Data Centers tab's model. Non-US "
                           "entities borrow the US pace, widened by any "
                           "disagreement with their own."),
        *([("its largest networkable group",
            "Unfiltered by host. *China-accessible* adds Chinese labs' sites "
            "abroad (DayOne Johor); *domestic only* is the mainland alone.")]
          if party == 'country' else []))

    # ── Table ──
    def _fmt(d, alt="—"):
        return f"{d:%b %Y}" if d is not None else alt

    table = []
    for r in recs:
        if r['crossed']:
            med = (f"crossed {_fmt(r['plan'])}"
                   + (f" with {r['via']}" if r['via'] else ""))
            rng = "—"
        else:
            med = _fmt(r['med'], f">{pc_horizon.year}")
            rng = (f"{_fmt(r['lo'])} – "
                   f"{_fmt(r['hi'], f'>{pc_horizon.year}')}"
                   if r['lo'] is not None else "—")
        table.append({
            "Entity": r['label'],
            "Plan crosses": _fmt(r['plan']),
            "Projected (median)": med,
            "80% range": rng,
            **{f"P(EOY {yr})": f"{r['p_by'][yr]:.0%}"
               for yr in _PC_TABLE_YEARS},
        })
    st.table(table)
    if timing_label == "Data center construction":
        _when = ("Dates are when the capacity is online; one run finishes "
                 f"{run_days // 30} months later.")
    elif timing_label == "Training run finished":
        _when = (f"Dates include the {run_days // 30}-month run: when a "
                 "model of this scale first finishes training.")
    else:
        _when = (f"Dates include the {run_days // 30}-month run plus "
                 "30 days of release prep.")
    _fn_caption(
        "<i>Plan crosses</i> reads the catalogue at face value; the projection "
        f"adds slip and a trend. {_when} Source: [Epoch AI, Frontier Data "
        "Centers](https://epoch.ai/data/data-centers) (CC-BY).",
        ("adds slip and a trend", "Timing slip on catalogued plans, then a "
                                  "fitted trend beyond them."))


def _render_anchor_links(scroll_to, tab_slug):
    """Make heading anchors shareable, and honour `?to=` on arrival.

    Streamlit's own link icon offers a bare `#fragment`, which is dropped
    before the heading exists (see the `?to=` note above), so each main-column
    heading link is rewritten to an absolute `?tab=…&to=…` URL of the *outer*
    page — the one a reader would share — while the click itself stays an
    in-page scroll. Arriving with `?to=` polls for the heading, since Streamlit
    streams the page in and Plotly resizes after insert; the re-settle stops on
    the first sign of the reader scrolling for themselves.
    """
    st.html(
        """
        <script>
        (function () {
          var TAB = "%s";
          var TO = "%s";

          function root() {
            return document.querySelector('[data-testid="stMain"]') || document.body;
          }
          function find(id) {
            return root().querySelector('[id="' + CSS.escape(id) + '"]')
                   || document.getElementById(id);
          }
          function shareUrl(id) {
            var loc;
            try { loc = window.parent.location; } catch (e) { loc = window.location; }
            var u;
            try { u = new URL(loc.href); } catch (e) { u = new URL(window.location.href); }
            u.hash = "";
            if (TAB) { u.searchParams.set("tab", TAB); }
            u.searchParams.set("to", id);
            return u.toString();
          }
          function relabel() {
            var sel = "h1 a[href^='#'], h2 a[href^='#'], h3 a[href^='#'], " +
                      "h4 a[href^='#'], h5 a[href^='#'], h6 a[href^='#']";
            root().querySelectorAll(sel).forEach(function (a) {
              var id = decodeURIComponent(a.getAttribute("href").slice(1));
              if (!id) { return; }
              a.href = shareUrl(id);
              if (a.dataset.vpAnchor) { return; }
              a.dataset.vpAnchor = "1";
              a.addEventListener("click", function (ev) {
                ev.preventDefault();
                var el = find(id);
                if (el) { el.scrollIntoView({block: "start"}); }
              });
            });
          }
          function watch() {
            relabel();
            new MutationObserver(relabel).observe(root(), {childList: true, subtree: true});
          }
          function jump(tries) {
            var el = find(TO);
            if (!el) {
              if (tries < 150) { setTimeout(function () { jump(tries + 1); }, 100); }
              return;
            }
            el.scrollIntoView({block: "start"});
            var n = 0;
            var timer = setInterval(function () {
              var e = find(TO);
              if (e) { e.scrollIntoView({block: "start"}); }
              if (++n >= 12) { stop(); }
            }, 250);
            function stop() {
              clearInterval(timer);
              ["wheel", "touchstart", "keydown"].forEach(function (ev) {
                window.removeEventListener(ev, stop, true);
              });
            }
            ["wheel", "touchstart", "keydown"].forEach(function (ev) {
              window.addEventListener(ev, stop, true);
            });
          }

          if (!window.__vpAnchorWatch) { window.__vpAnchorWatch = true; watch(); }
          else { relabel(); }
          if (TO && !window.__vpAnchorJumped) { window.__vpAnchorJumped = true; jump(0); }
        })();
        </script>
        """ % (tab_slug, scroll_to),
        unsafe_allow_javascript=True,
    )


# ── Dispatch ─────────────────────────────────────────────────────────────

if not os.environ.get("_VP_TESTING"):
    _hydrate_session_from_url()

    if active_tab == "METR Horizon":
        render_metr()
    elif active_tab == "Epoch ECI":
        render_eci()
    elif active_tab == "Remote Labor Index":
        render_rli()
    elif active_tab == "RSI":
        render_rsi()
    elif active_tab == "UK Cyber":
        render_ukcyber()
    elif active_tab == "Revenue":
        render_revenue()
    elif active_tab == "Employment":
        render_employment()
    elif active_tab == "ECI Company Gap":
        render_eci_gap()
    elif active_tab == "Data Centers":
        render_data_centers()
    elif active_tab == "Compute/capabilities/diffusion":
        render_compute_capabilities()
    elif active_tab == "Pacing":
        render_pacing()

    _sync_session_to_url()

    # ── Section deep links: honour ?to=, and make every heading emit one ──
    _render_anchor_links(_url_to, _SLUG_FOR_TAB[active_tab])

    # ── Share view: copy the (now state-synced) URL to the clipboard ──────
    with st.sidebar:
        st.markdown("---")
        st.html(
            """
            <button id="share-view-btn" onclick="copyShareView(this)">
              🔗 Share view
            </button>
            <style>
              #share-view-btn {
                width: 100%;
                padding: 0.5rem 0.75rem;
                font-size: 0.875rem;
                font-weight: 600;
                font-family: "Source Sans Pro", sans-serif;
                color: rgb(49, 51, 63);
                background-color: #fff;
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                cursor: pointer;
                transition: all 0.15s ease;
              }
              #share-view-btn:hover {
                border-color: rgb(255, 75, 75);
                color: rgb(255, 75, 75);
              }
              #share-view-btn.copied {
                border-color: rgb(33, 195, 84);
                color: rgb(33, 195, 84);
              }
            </style>
            <script>
              function _flash(btn, text) {
                const orig = btn.innerHTML;
                btn.innerHTML = text;
                btn.classList.add("copied");
                setTimeout(function () {
                  btn.innerHTML = orig;
                  btn.classList.remove("copied");
                }, 1500);
              }
              function _fallbackCopy(url, btn) {
                const ta = document.createElement("textarea");
                ta.value = url;
                ta.style.position = "fixed";
                ta.style.opacity = "0";
                document.body.appendChild(ta);
                ta.select();
                try { document.execCommand("copy"); _flash(btn, "✓ Copied!"); }
                catch (e) { _flash(btn, "⚠ Copy failed"); }
                document.body.removeChild(ta);
              }
              function copyShareView(btn) {
                let url;
                try { url = window.parent.location.href; }
                catch (e) { url = document.referrer || window.location.href; }
                if (navigator.clipboard && navigator.clipboard.writeText) {
                  navigator.clipboard.writeText(url).then(
                    function () { _flash(btn, "✓ Copied!"); },
                    function () { _fallbackCopy(url, btn); }
                  );
                } else {
                  _fallbackCopy(url, btn);
                }
              }
            </script>
            """,
            unsafe_allow_javascript=True,
        )
