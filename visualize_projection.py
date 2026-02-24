"""
AI Capability Projections: Interactive Plotly fan charts for METR Horizon and Epoch ECI benchmarks.
Run: streamlit run visualize_projection.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml
import csv
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AI Capability Projections", layout="wide")

# Tighten default Streamlit padding
st.markdown("""<style>
    .block-container { padding-top: 2rem !important; }
    [data-testid="stTable"] table { margin-top: 0 !important; margin-bottom: 0.5rem !important; }
</style>""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────

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
}


def pretty(name):
    return _NAMES.get(name, name)


def log2min_to_label(val):
    """Convert log2(minutes) to human-readable string."""
    minutes = 2 ** val
    if minutes < 1:
        return f"{minutes*60:.0f}s"
    if minutes < 60:
        return f"{minutes:.0f}m"
    hrs = minutes / 60
    return fmt_hrs(hrs)


def fmt_hrs(h):
    """Format hours for display using work-time units (8h/d, 40h/w, 176h/mo, 2000h/y).
    Shows sub-unit remainder (e.g., 1h20m, 2d3h). No decimals."""
    minutes = h * 60
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
def load_frontier(_mtime=None):
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
    frontier = [m for m in models if m['is_sota']]
    return frontier


def _eci_mtime():
    csv_path = os.path.join(os.path.dirname(__file__), 'epoch_capabilities_index.csv')
    return os.path.getmtime(csv_path)


@st.cache_data
def load_eci_frontier(_mtime=None):
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
        })

    # Dedup: keep highest-scoring variant per model name
    best_by_name = {}
    for m in valid:
        name = m['name']
        if name not in best_by_name or m['eci_score'] > best_by_name[name]['eci_score']:
            best_by_name[name] = m
    deduped = sorted(best_by_name.values(), key=lambda m: m['date'])

    # Filter to Claude 3 Opus era onward (Feb 2024+)
    _cutoff_date = datetime(2024, 2, 29)
    deduped = [m for m in deduped if m['date'] >= _cutoff_date]

    # Frontier detection: running max
    max_score = -float('inf')
    for m in deduped:
        if m['eci_score'] > max_score:
            max_score = m['eci_score']
            m['is_frontier'] = True
        else:
            m['is_frontier'] = False

    return deduped


# ── RLI data (hardcoded – small dataset from remotelabor.ai) ─────────────

_RLI_RAW = [
    {"name": "Gemini 2.5 Pro", "date": "2025-03-25", "rli_score": 0.83},
    {"name": "Grok 4",         "date": "2025-07-01", "rli_score": 2.10},
    {"name": "GPT-5",          "date": "2025-08-07", "rli_score": 1.67},
    {"name": "Sonnet 4.5",     "date": "2025-09-20", "rli_score": 2.08},
    {"name": "Manus 1.5",      "date": "2025-10-20", "rli_score": 2.50},
    {"name": "Opus 4.5",       "date": "2025-11-15", "rli_score": 3.75},
    {"name": "Gemini 3 Pro",   "date": "2025-12-10", "rli_score": 1.25},
    {"name": "GPT-5.2",        "date": "2025-12-20", "rli_score": 2.50},
]


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


# ── Load data (before sidebar, so model names are available) ─────────────

frontier_all = load_frontier(_mtime=_yaml_mtime())
gpt4o_idx = next(i for i, m in enumerate(frontier_all) if m['name'] == 'gpt_4o_inspect')
frontier_names = [pretty(m['name']) for m in frontier_all]

eci_all = load_eci_frontier(_mtime=_eci_mtime())
eci_frontier_all = [m for m in eci_all if m['is_frontier']]
eci_frontier_names = [m['display_name'] for m in eci_frontier_all]

rli_all = load_rli_data()
rli_frontier_all = [m for m in rli_all if m['is_frontier']]
rli_frontier_names = [m['name'] for m in rli_frontier_all]


# ── Sidebar: tab selector ────────────────────────────────────────────────

_TAB_OPTIONS = ["METR Horizon", "Epoch ECI", "Remote Labor Index"]
_TAB_SLUG = {"metr": 0, "eci": 1, "rli": 2}

# Read ?tab= from URL for deep-linking
_url_tab = st.query_params.get("tab", "").lower()
_default_tab_idx = _TAB_SLUG.get(_url_tab, 0)

with st.sidebar:
    active_tab = st.radio("Tab", _TAB_OPTIONS, index=_default_tab_idx, horizontal=True, key="_active_tab")
    st.markdown("---")

# Keep URL in sync with selected tab
_SLUG_FOR_TAB = {"METR Horizon": "metr", "Epoch ECI": "eci", "Remote Labor Index": "rli"}
st.query_params["tab"] = _SLUG_FOR_TAB[active_tab]



# ── METR Horizon ─────────────────────────────────────────────────────────

def render_metr():
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
        proj_basis = st.radio("Projection basis", basis_options, index=1)

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
        if _is_linear:
            with st.expander("Advanced options"):
                custom_dt_lo, custom_dt_hi = st.columns(2)
                custom_dt_lo = custom_dt_lo.number_input(
                    "DT CI low (days)", value=55,
                    min_value=10, max_value=2000, step=5, key="custom_dt_lo")
                custom_dt_hi = custom_dt_hi.number_input(
                    "DT CI high (days)", value=180,
                    min_value=10, max_value=2000, step=5, key="custom_dt_hi")
                if custom_dt_lo > custom_dt_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()

                _cur = frontier_all[proj_as_of_idx]
                _def_lo_hrs = (_cur.get(_sb_lo_key) or _cur[_sb_val_key]) / 60
                _def_hi_hrs = (_cur.get(_sb_hi_key) or _cur[_sb_val_key]) / 60
                _pos_lo_col, _pos_hi_col = st.columns(2)
                _p_suffix = "_p80" if _sidebar_p80 else "_p50"
                custom_pos_lo = _pos_lo_col.number_input(
                    "Pos CI low (h)", value=round(_def_lo_hrs, 1),
                    min_value=0.01, step=0.5, key="custom_pos_lo" + _p_suffix)
                custom_pos_hi = _pos_hi_col.number_input(
                    "Pos CI high (h)", value=round(_def_hi_hrs, 1),
                    min_value=0.01, step=0.5, key="custom_pos_hi" + _p_suffix)

                piecewise_n_segments = st.radio(
                    "Segments", [1, 2, 3],
                    index={1: 0, 2: 1, 3: 2}[piecewise_n_segments],
                    horizontal=True, key="piecewise_n_seg")
                _bp_names = [pretty(m['name']) for m in frontier_all[:proj_as_of_idx + 1]]
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
                    if len(_remaining) >= 2:
                        bp2_name = st.selectbox(
                            "Breakpoint 2", _remaining[:-1],
                            index=len(_remaining[:-1]) // 2, key="bp2_select")
                        piecewise_breakpoints.append(bp2_name)
                    else:
                        st.warning("Not enough models for 3 segments.")
                        piecewise_n_segments = 2

                custom_dt_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"], index=1,
                    horizontal=True, key="custom_dt_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space. "
                         "Log-log: fat right tail.")
                custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal", "Log-log"], index=1,
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

            with st.expander("Advanced options"):
                _se_col1, _se_col2 = st.columns(2)
                superexp_dt_initial = _se_col1.number_input(
                    "Initial DT (days)", value=_default_dt_init,
                    min_value=10, max_value=2000, step=5, key="superexp_dt_init")
                superexp_halflife = _se_col2.number_input(
                    "DT half-life (days)", value=365,
                    min_value=30, max_value=5000, step=30, key="superexp_halflife",
                    help="How quickly DT shrinks. Lower = faster.")
                superexp_dt_floor = st.number_input(
                    "Min DT floor (days)", value=30,
                    min_value=1, max_value=500, step=5, key="superexp_dt_floor",
                    help="DT can't shrink below this. Prevents runaway projections.")
                _se_ci1, _se_ci2 = st.columns(2)
                superexp_dt_ci_lo = _se_ci1.number_input(
                    "DT CI low (days)", value=80,
                    min_value=10, max_value=2000, step=5, key="superexp_dt_ci_lo")
                superexp_dt_ci_hi = _se_ci2.number_input(
                    "DT CI high (days)", value=250,
                    min_value=10, max_value=2000, step=5, key="superexp_dt_ci_hi")
                if superexp_dt_ci_lo > superexp_dt_ci_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()
                _cur = frontier_all[proj_as_of_idx]
                _def_lo_hrs = (_cur.get(_sb_lo_key) or _cur[_sb_val_key]) / 60
                _def_hi_hrs = (_cur.get(_sb_hi_key) or _cur[_sb_val_key]) / 60
                _se_pos1, _se_pos2 = st.columns(2)
                _p_suffix_se = "_p80" if _sidebar_p80 else "_p50"
                superexp_pos_lo = _se_pos1.number_input(
                    "Pos CI low (h)", value=round(_def_lo_hrs, 1),
                    min_value=0.01, step=0.5, key="superexp_pos_lo" + _p_suffix_se)
                superexp_pos_hi = _se_pos2.number_input(
                    "Pos CI high (h)", value=round(_def_hi_hrs, 1),
                    min_value=0.01, step=0.5, key="superexp_pos_hi" + _p_suffix_se)

        st.markdown("---")
        show_milestones = st.toggle("Milestones", value=True, key="milestones")
        show_labels = st.toggle("Labels", value=True, key="labels")
        only_post_gpt4o = st.toggle("GPT-4o+ only", value=False, key="post_gpt4o")
        use_p80 = st.toggle("Use p80", value=False, key="p80")
        use_log_scale = st.toggle("Log scale", value=True, key="log_scale")

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
                index=0, horizontal=True, key="metr_end_year")

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
        n_custom = 20000
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
        n_superexp = 20000
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
    all_trajectories = np.zeros((n_samples, len(proj_days_arr)))
    if is_superexp:
        halflife_val = superexp_halflife
        dt_floor = superexp_dt_floor
        for i in range(n_samples):
            # Time when DT hits the floor: dt_0 * 2^(-t_cap/H) = floor
            if proj_dt[i] > dt_floor:
                t_cap = halflife_val * np.log2(proj_dt[i] / dt_floor)
            else:
                t_cap = 0.0  # already at or below floor
            # Before t_cap: superexponential growth
            se_phase = np.minimum(proj_days_arr, t_cap)
            y_se = (halflife_val / (proj_dt[i] * np.log(2))) * (2**(se_phase / halflife_val) - 1)
            # After t_cap: linear growth at floor DT
            linear_phase = np.maximum(proj_days_arr - t_cap, 0)
            y_lin = linear_phase / dt_floor
            all_trajectories[i] = proj_start[i] + y_se + y_lin
    else:
        for i in range(n_samples):
            all_trajectories[i] = proj_start[i] + proj_days_arr / proj_dt[i]

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
            texts.append(f"{d.strftime('%b %d, %Y')}<br>Trend: {fmt_hrs(h)}")
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
                        mode='lines', line=dict(color='#1a5276', width=2.5),
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
                mode='lines', line=dict(color='#1a5276', width=2.5),
                name=f'Projection ({_user_dt_center:.0f}d doubling, CI {custom_dt_lo}\u2013{custom_dt_hi}d)',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif proj_basis == "Superexponential":
        # Reuse the fit computed earlier: y = _se_A + _se_K * 2^(d / halflife)
        d_start = int(days_used[0])
        d_end = (proj_end_date - base_date).days
        days_range = np.arange(d_start, d_end + 1, 1)
        y_log2_seg = _se_A_disp + _se_K * 2 ** (days_range / superexp_halflife)
        dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
        hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {fmt_hrs(2**y / 60)}" for dt, y in zip(dates_seg, y_log2_seg)]
        y_seg = _yconv(y_log2_seg)
        y_seg = y_seg.tolist() if hasattr(y_seg, 'tolist') else list(y_seg)
        fig.add_trace(go.Scatter(
            x=dates_seg, y=y_seg,
            mode='lines', line=dict(color='#8e44ad', width=2.5),
            name=f'Superexp fit (current DT\u2248{superexp_dt_fitted:.0f}d, half-life={superexp_halflife}d)',
            hovertext=hover_seg, hoverinfo='text',
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

    # --- Today vline ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

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
        tick_text = [log2min_to_label(v) for v in tick_vals]
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
    st.plotly_chart(fig, use_container_width=True)
    if is_backtesting and backtest_results:
        _backtest_summary(backtest_results)

    # ── Projections ───────────────────────────────────────────────────────────

    start_hrs_samples = 2**proj_start / 60
    med_dt = np.median(proj_dt)
    p10_dt, p90_dt = np.percentile(proj_dt, [10, 90])
    current_label = pretty(current['name'])

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026 Jun EOM", datetime(2026, 6, 30)),
        ("2026EOY", datetime(2026, 12, 31)),
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
            st.metric(label=label, value=fmt_hrs(display_h))
            st.caption(f"80% CI: {fmt_hrs(p10_h)} \u2013 {fmt_hrs(p90_h)}")

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

    st.caption("Time units are human work-time: 1d = 8h, 1w = 40h, 1mo = 176h, 1y = 2000h.")


# ── Epoch ECI ────────────────────────────────────────────────────────────

def render_eci():
    # ── ECI Sidebar controls ─────────────────────────────────────────────
    with st.sidebar:
        st.header("ECI Projection")

        # Read "project as of" from session state
        eci_proj_as_of_name = st.session_state.get('_eci_proj_as_of', eci_frontier_names[-1])
        if eci_proj_as_of_name not in eci_frontier_names:
            eci_proj_as_of_name = eci_frontier_names[-1]
        eci_proj_as_of_idx = eci_frontier_names.index(eci_proj_as_of_name)

        # --- Projection basis ---
        eci_basis_options = ["Linear", "Piecewise linear", "Superexponential"]
        eci_proj_basis = st.radio("Projection basis", eci_basis_options, index=0, key="eci_proj_basis")

        eci_custom_dpp_lo = eci_custom_dpp_hi = None
        eci_custom_pos_lo = eci_custom_pos_hi = None
        eci_custom_dpp_dist = "Lognormal"
        eci_custom_pos_dist = "Lognormal"
        eci_piecewise_n_segments = 1
        eci_piecewise_breakpoints = []
        _eci_is_linear = eci_proj_basis in ("Linear", "Piecewise linear")
        if eci_proj_basis == "Piecewise linear":
            eci_piecewise_n_segments = 2

        if _eci_is_linear:
            with st.expander("Advanced options"):
                _eci_ppy_lo_col, _eci_ppy_hi_col = st.columns(2)
                eci_custom_ppy_lo = _eci_ppy_lo_col.number_input(
                    "+Pts/Yr CI low", value=7.0,
                    min_value=0.5, max_value=365.0, step=0.5, key="eci_custom_ppy_lo")
                eci_custom_ppy_hi = _eci_ppy_hi_col.number_input(
                    "+Pts/Yr CI high", value=18.0,
                    min_value=0.5, max_value=365.0, step=0.5, key="eci_custom_ppy_hi")
                if eci_custom_ppy_lo > eci_custom_ppy_hi:
                    st.error("+Pts/Yr CI low must be ≤ +Pts/Yr CI high.")
                    st.stop()
                eci_custom_dpp_lo = 365.25 / eci_custom_ppy_hi  # high PPY = low DPP (fast)
                eci_custom_dpp_hi = 365.25 / eci_custom_ppy_lo  # low PPY = high DPP (slow)

                # Position CI: fitted score +/- 2
                _eci_cur = eci_frontier_all[eci_proj_as_of_idx]
                _eci_def_score = _eci_cur['eci_score']
                _eci_pos_lo_col, _eci_pos_hi_col = st.columns(2)
                eci_custom_pos_lo = _eci_pos_lo_col.number_input(
                    "Pos CI low (ECI)", value=round(_eci_def_score - 2, 1),
                    step=0.5, key="eci_custom_pos_lo")
                eci_custom_pos_hi = _eci_pos_hi_col.number_input(
                    "Pos CI high (ECI)", value=round(_eci_def_score + 2, 1),
                    step=0.5, key="eci_custom_pos_hi")

                eci_piecewise_n_segments = st.radio(
                    "Segments", [1, 2, 3],
                    index={1: 0, 2: 1, 3: 2}[eci_piecewise_n_segments],
                    horizontal=True, key="eci_piecewise_n_seg")
                _eci_bp_names = [m['display_name'] for m in eci_frontier_all[:eci_proj_as_of_idx + 1]]
                if eci_piecewise_n_segments >= 2:
                    _eci_default_bp1 = _eci_bp_names[len(_eci_bp_names) // 2]
                    _eci_bp1_idx = _eci_bp_names.index(_eci_default_bp1) if _eci_default_bp1 in _eci_bp_names else len(_eci_bp_names) // 2
                    eci_bp1_name = st.selectbox(
                        "Breakpoint", _eci_bp_names[1:],
                        index=max(0, _eci_bp1_idx - 1), key="eci_bp1_select")
                    eci_piecewise_breakpoints.append(eci_bp1_name)
                if eci_piecewise_n_segments >= 3:
                    _eci_bp1_pos = _eci_bp_names.index(eci_bp1_name)
                    _eci_remaining = _eci_bp_names[_eci_bp1_pos + 1:]
                    if len(_eci_remaining) >= 2:
                        eci_bp2_name = st.selectbox(
                            "Breakpoint 2", _eci_remaining[:-1],
                            index=len(_eci_remaining[:-1]) // 2, key="eci_bp2_select")
                        eci_piecewise_breakpoints.append(eci_bp2_name)
                    else:
                        st.warning("Not enough models for 3 segments.")
                        eci_piecewise_n_segments = 2

                eci_custom_dpp_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"], index=1,
                    horizontal=True, key="eci_custom_dpp_dist",
                    help="Normal: symmetric. Lognormal: symmetric in log-space. "
                         "Log-log: fat right tail.")
                eci_custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal"], index=0,
                    horizontal=True, key="eci_custom_pos_dist",
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
            if len(eci_frontier_all[:eci_proj_as_of_idx + 1]) >= 2:
                _eci_base = eci_frontier_all[0]['date']
                _eci_fr = eci_frontier_all[:eci_proj_as_of_idx + 1]
                _eci_fd = np.array([(m['date'] - _eci_base).days for m in _eci_fr], dtype=float)
                _eci_fs = np.array([m['eci_score'] for m in _eci_fr])
                _eci_fp = fit_line(_eci_fd, _eci_fs)
                if _eci_fp[1] > 0:
                    _eci_default_ppy_init = round(365.25 * _eci_fp[1], 1)

            with st.expander("Advanced options"):
                _eci_se_col1, _eci_se_col2 = st.columns(2)
                eci_superexp_ppy_initial = _eci_se_col1.number_input(
                    "Initial +Pts/Yr", value=_eci_default_ppy_init,
                    min_value=0.5, max_value=365.0, step=0.5, key="eci_superexp_ppy_init")
                eci_superexp_dpp_initial = 365.25 / eci_superexp_ppy_initial
                eci_superexp_halflife = _eci_se_col2.number_input(
                    "Rate half-life (days)", value=365,
                    min_value=30, max_value=5000, step=30, key="eci_superexp_halflife",
                    help="How quickly rate grows. Lower = faster.")
                eci_superexp_ppy_ceiling = st.number_input(
                    "Max +Pts/Yr ceiling", value=37.0,
                    min_value=1.0, max_value=365.0, step=1.0, key="eci_superexp_ppy_ceiling",
                    help="Rate can't exceed this. Prevents runaway projections.")
                eci_superexp_dpp_floor = 365.25 / eci_superexp_ppy_ceiling
                _eci_se_ci1, _eci_se_ci2 = st.columns(2)
                eci_superexp_ppy_ci_lo = _eci_se_ci1.number_input(
                    "+Pts/Yr CI low", value=6.0,
                    min_value=0.5, max_value=365.0, step=0.5, key="eci_superexp_ppy_ci_lo")
                eci_superexp_ppy_ci_hi = _eci_se_ci2.number_input(
                    "+Pts/Yr CI high", value=24.0,
                    min_value=0.5, max_value=365.0, step=0.5, key="eci_superexp_ppy_ci_hi")
                if eci_superexp_ppy_ci_lo > eci_superexp_ppy_ci_hi:
                    st.error("+Pts/Yr CI low must be ≤ +Pts/Yr CI high.")
                    st.stop()
                eci_superexp_dpp_ci_lo = 365.25 / eci_superexp_ppy_ci_hi  # high PPY = low DPP
                eci_superexp_dpp_ci_hi = 365.25 / eci_superexp_ppy_ci_lo  # low PPY = high DPP
                _eci_cur = eci_frontier_all[eci_proj_as_of_idx]
                _eci_def_score = _eci_cur['eci_score']
                _eci_se_pos1, _eci_se_pos2 = st.columns(2)
                eci_superexp_pos_lo = _eci_se_pos1.number_input(
                    "Pos CI low (ECI)", value=round(_eci_def_score - 2, 1),
                    step=0.5, key="eci_superexp_pos_lo")
                eci_superexp_pos_hi = _eci_se_pos2.number_input(
                    "Pos CI high (ECI)", value=round(_eci_def_score + 2, 1),
                    step=0.5, key="eci_superexp_pos_hi")

        st.markdown("---")
        eci_show_milestones = st.toggle("Milestones", value=True, key="eci_milestones")
        eci_show_labels = st.toggle("Labels", value=True, key="eci_labels")

        st.markdown("---")
        with st.expander("Projection range"):
            st.selectbox(
                "Project as of",
                eci_frontier_names,
                index=eci_frontier_names.index(eci_proj_as_of_name),
                key='_eci_proj_as_of',
                help="Backtest: project from an earlier model's vantage point.",
            )
            _eci_end_year = st.radio(
                "Project through", [2026, 2027, 2028, 2029],
                index=0, horizontal=True, key="eci_end_year")

    # ── Build data arrays ────────────────────────────────────────────────────
    eci_frontier_used = eci_frontier_all[:eci_proj_as_of_idx + 1]
    eci_frontier_plot = list(eci_all)  # show all models (frontier + non-frontier)

    base_date = eci_frontier_all[0]['date']
    days_all_eci = np.array([(m['date'] - base_date).days for m in eci_frontier_all], dtype=float)
    scores_all_eci = np.array([m['eci_score'] for m in eci_frontier_all])

    _eci_fit_start = 0
    _eci_fit_end = eci_proj_as_of_idx + 1
    eci_frontier_used = eci_frontier_all[_eci_fit_start:_eci_fit_end]
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

        n_eci = 20000
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

        n_eci = 20000
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
    all_trajectories = np.zeros((n_samples, len(proj_days_arr)))
    if eci_is_superexp:
        halflife_val = eci_superexp_halflife
        dpp_floor = eci_superexp_dpp_floor
        for i in range(n_samples):
            # For ECI (linear score): points gained = integral of 1/dpp(t) dt
            # dpp(t) = dpp_0 * 2^(-t/H), so points = (H/(dpp_0*ln2)) * (2^(t/H) - 1)
            # After DPP hits floor: linear at floor rate
            if eci_proj_dpp[i] > dpp_floor:
                t_cap = halflife_val * np.log2(eci_proj_dpp[i] / dpp_floor)
            else:
                t_cap = 0.0
            se_phase = np.minimum(proj_days_arr, t_cap)
            pts_se = (halflife_val / (eci_proj_dpp[i] * np.log(2))) * (2**(se_phase / halflife_val) - 1)
            linear_phase = np.maximum(proj_days_arr - t_cap, 0)
            pts_lin = linear_phase / dpp_floor
            all_trajectories[i] = eci_proj_start[i] + pts_se + pts_lin
    else:
        for i in range(n_samples):
            all_trajectories[i] = eci_proj_start[i] + proj_days_arr / eci_proj_dpp[i]

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

    # --- Trend lines ---
    def _eci_trend_hover(params, d_start, d_end, base_dt):
        """Build hover texts for an OLS trend line on ECI scores."""
        days_range = np.arange(d_start, d_end + 1, 1)
        dates = [base_dt + timedelta(days=int(d)) for d in days_range]
        y_scores = params[0] + params[1] * days_range
        texts = []
        for d, y in zip(dates, y_scores):
            texts.append(f"{d.strftime('%b %d, %Y')}<br>Trend: {y:.1f}")
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
                        mode='lines', line=dict(color='#1a5276', width=2.5),
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
                mode='lines', line=dict(color='#1a5276', width=2.5),
                name=f'Projection ({_user_ppy_center:.1f} pts/yr, CI {eci_custom_ppy_lo}\u2013{eci_custom_ppy_hi})',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif eci_proj_basis == "Superexponential":
        d_start = int(days_used[0])
        d_end = (proj_end_date - base_date).days
        days_range = np.arange(d_start, d_end + 1, 1)
        y_scores_seg = _eci_se_A + _eci_se_K * 2 ** (days_range / eci_superexp_halflife)
        dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
        hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.1f}" for dt, y in zip(dates_seg, y_scores_seg)]
        fig.add_trace(go.Scatter(
            x=dates_seg, y=y_scores_seg.tolist(),
            mode='lines', line=dict(color='#8e44ad', width=2.5),
            name=f'Superexp fit ({365.25/eci_superexp_dpp_fitted:.1f} pts/yr, half-life={eci_superexp_halflife}d)',
            hovertext=hover_seg, hoverinfo='text',
        ))

    # --- Milestone hlines ---
    if eci_show_milestones:
        x_lo = eci_all[0]['date'] - timedelta(days=30)
        x_hi = proj_end_date
        for score_val, label, color in [
            (155, "ECI 155", '#888888'),
            (160, "ECI 160", '#666666'),
            (165, "ECI 165", '#c0392b'),
            (170, "ECI 170", '#8e44ad'),
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

    # --- Today vline ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

    # --- Backtesting ---
    eci_is_backtesting = eci_proj_as_of_idx < len(eci_frontier_all) - 1
    eci_backtest_results = []
    _eci_bt_lookup = {}
    if eci_is_backtesting:
        _eci_bt_future = eci_frontier_all[eci_proj_as_of_idx + 1:]
        eci_backtest_results = _backtest_stats(
            _eci_bt_future, all_trajectories, eci_current['date'], proj_end_date,
            lambda m: m['eci_score'],
            lambda m: m['display_name'],
        )
        _eci_bt_lookup = {r['name']: r for r in eci_backtest_results}

    # --- Data points ---
    # Non-frontier models: only show those within 10 pts of frontier max to reduce clutter
    _eci_frontier_max = max(m['eci_score'] for m in eci_all if m['is_frontier'])
    _eci_nf_cutoff = _eci_frontier_max - 10
    for m in eci_all:
        if m['is_frontier'] or m['eci_score'] < _eci_nf_cutoff:
            continue
        hover = f"{m['display_name']}<br>{m['date'].strftime('%b %d, %Y')}<br>ECI: {m['eci_score']:.1f}"
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
    for idx_m, m in enumerate(eci_frontier_all):
        is_used = idx_m <= eci_proj_as_of_idx
        is_selected = idx_m == eci_proj_as_of_idx
        hover = f"{m['display_name']}<br>{m['date'].strftime('%b %d, %Y')}<br>ECI: {m['eci_score']:.1f}"

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
    all_scores = [m['eci_score'] for m in eci_all if m['is_frontier'] or m['eci_score'] >= _eci_nf_cutoff]
    y_min = min(all_scores) - 5
    y_max = max(pct95[-1], max(all_scores) + 5, 170) + 5
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
            range=[eci_all[0]['date'] - timedelta(days=30),
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
    st.plotly_chart(fig, use_container_width=True)
    if eci_is_backtesting and eci_backtest_results:
        _backtest_summary(eci_backtest_results)

    # ── Projections row ───────────────────────────────────────────────────

    eci_start_samples = eci_proj_start
    eci_current_label = eci_current['display_name']

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026 Jun EOM", datetime(2026, 6, 30)),
        ("2026EOY", datetime(2026, 12, 31)),
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
    for col, (label, target_date) in zip(cols, all_targets):
        elapsed = (target_date - eci_current['date']).days
        proj_scores = _proj_score_at(
            elapsed, eci_start_samples, eci_proj_dpp,
            eci_is_superexp, eci_superexp_halflife,
            eci_superexp_dpp_floor if eci_is_superexp else None)
        p10_s, p50_s, p90_s = np.percentile(proj_scores, [10, 50, 90])
        display_s = eci_current_score if elapsed == 0 else p50_s
        with col:
            st.metric(label=label, value=f"{display_s:.1f}")
            st.caption(f"80% CI: {p10_s:.1f} \u2013 {p90_s:.1f}")

    # Milestone tables
    eci_milestone_thresholds = [
        (155, "ECI 155"),
        (160, "ECI 160"),
        (165, "ECI 165"),
        (170, "ECI 170"),
    ]

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

    st.caption("ECI = Epoch Capabilities Index. +Pts/Yr = ECI points gained per year.")


# ── Remote Labor Index ───────────────────────────────────────────────────

def render_rli():
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
        rli_proj_basis = st.radio("Projection basis", rli_basis_options, index=0, key="rli_proj_basis",
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

        if _rli_is_linear:
            with st.expander("Advanced options"):
                # Doubling time CI (days for odds to double)
                _rli_dt_lo_col, _rli_dt_hi_col = st.columns(2)
                rli_custom_dt_lo = _rli_dt_lo_col.number_input(
                    "Odds 2x time CI low (days)", value=100.0,
                    min_value=5.0, max_value=2000.0, step=5.0, key="rli_custom_dt_lo",
                    help="Fast scenario: days for odds p/(1-p) to double.")
                rli_custom_dt_hi = _rli_dt_hi_col.number_input(
                    "Odds 2x time CI high (days)", value=200.0,
                    min_value=5.0, max_value=5000.0, step=5.0, key="rli_custom_dt_hi",
                    help="Slow scenario: days for odds to double.")
                if rli_custom_dt_lo > rli_custom_dt_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()

                # Position CI in percentage points
                _rli_cur = rli_frontier_all[rli_proj_as_of_idx]
                _rli_def_score = _rli_cur['rli_score']
                _rli_pos_lo_col, _rli_pos_hi_col = st.columns(2)
                rli_custom_pos_lo = _rli_pos_lo_col.number_input(
                    "Pos CI low (%)", value=round(max(_rli_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1, key="rli_custom_pos_lo")
                rli_custom_pos_hi = _rli_pos_hi_col.number_input(
                    "Pos CI high (%)", value=round(_rli_def_score + 1.0, 2),
                    step=0.1, key="rli_custom_pos_hi")

                rli_piecewise_n_segments = st.radio(
                    "Segments", [1, 2, 3],
                    index={1: 0, 2: 1, 3: 2}[rli_piecewise_n_segments],
                    horizontal=True, key="rli_piecewise_n_seg")
                _rli_bp_names = [m['name'] for m in rli_frontier_all[:rli_proj_as_of_idx + 1]]
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
                    if len(_rli_remaining) >= 2:
                        rli_bp2_name = st.selectbox(
                            "Breakpoint 2", _rli_remaining[:-1],
                            index=len(_rli_remaining[:-1]) // 2, key="rli_bp2_select")
                        rli_piecewise_breakpoints.append(rli_bp2_name)
                    else:
                        st.warning("Not enough models for 3 segments.")
                        rli_piecewise_n_segments = 2

                rli_custom_dt_dist = st.radio(
                    "Trend distribution", ["Normal", "Lognormal", "Log-log"], index=1,
                    horizontal=True, key="rli_custom_dt_dist")
                rli_custom_pos_dist = st.radio(
                    "Position distribution", ["Normal", "Lognormal"], index=0,
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

            with st.expander("Advanced options"):
                _rli_se_col1, _rli_se_col2 = st.columns(2)
                rli_superexp_dt_initial = _rli_se_col1.number_input(
                    "Initial odds 2x time (days)", value=_rli_default_dt_init,
                    min_value=5.0, max_value=2000.0, step=5.0, key="rli_superexp_dt_init")
                rli_superexp_halflife = _rli_se_col2.number_input(
                    "Rate half-life (days)", value=365,
                    min_value=30, max_value=5000, step=30, key="rli_superexp_halflife",
                    help="How quickly rate grows. Lower = faster.")
                rli_superexp_dt_floor_input = st.number_input(
                    "Min odds 2x time (days)", value=15.0,
                    min_value=1.0, max_value=500.0, step=1.0, key="rli_superexp_dt_floor",
                    help="Rate can't exceed this. Prevents runaway projections.")
                rli_superexp_dt_floor = rli_superexp_dt_floor_input
                _rli_se_ci1, _rli_se_ci2 = st.columns(2)
                rli_superexp_dt_ci_lo = _rli_se_ci1.number_input(
                    "Odds 2x CI low (days)", value=100.0,
                    min_value=5.0, max_value=2000.0, step=5.0, key="rli_superexp_dt_ci_lo")
                rli_superexp_dt_ci_hi = _rli_se_ci2.number_input(
                    "Odds 2x CI high (days)", value=200.0,
                    min_value=5.0, max_value=5000.0, step=5.0, key="rli_superexp_dt_ci_hi")
                if rli_superexp_dt_ci_lo > rli_superexp_dt_ci_hi:
                    st.error("DT CI low must be ≤ DT CI high.")
                    st.stop()
                _rli_cur = rli_frontier_all[rli_proj_as_of_idx]
                _rli_def_score = _rli_cur['rli_score']
                _rli_se_pos1, _rli_se_pos2 = st.columns(2)
                rli_superexp_pos_lo = _rli_se_pos1.number_input(
                    "Pos CI low (%)", value=round(max(_rli_def_score - 1.0, 0.1), 2),
                    min_value=0.01, step=0.1, key="rli_superexp_pos_lo")
                rli_superexp_pos_hi = _rli_se_pos2.number_input(
                    "Pos CI high (%)", value=round(_rli_def_score + 1.0, 2),
                    step=0.1, key="rli_superexp_pos_hi")

        st.markdown("---")
        rli_show_milestones = st.toggle("Milestones", value=True, key="rli_milestones")
        rli_show_labels = st.toggle("Labels", value=True, key="rli_labels")
        rli_use_log_scale = st.toggle("Log scale", value=False, key="rli_log_scale")

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
                index=0, horizontal=True, key="rli_end_year")

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

        n_rli = 20000
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

        n_rli = 20000
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
    all_logit_traj = np.zeros((n_samples, len(proj_days_arr)))
    if rli_is_superexp:
        halflife_val = rli_superexp_halflife
        # logit_slope(t) = slope_0 * 2^(t/H), floored at max slope (= ln2/dt_floor)
        slope_floor = np.log(2) / rli_superexp_dt_floor
        for i in range(n_samples):
            slope_0 = rli_proj_logit_slope[i]
            if slope_0 < slope_floor:
                t_cap = halflife_val * np.log2(slope_floor / slope_0)
            else:
                t_cap = 0.0
            se_phase = np.minimum(proj_days_arr, t_cap)
            logit_se = (halflife_val / np.log(2)) * slope_0 * (2**(se_phase / halflife_val) - 1)
            linear_phase = np.maximum(proj_days_arr - t_cap, 0)
            logit_lin = linear_phase * slope_floor
            all_logit_traj[i] = rli_proj_start_logit[i] + logit_se + logit_lin
    else:
        for i in range(n_samples):
            all_logit_traj[i] = rli_proj_start_logit[i] + proj_days_arr * rli_proj_logit_slope[i]

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
                        mode='lines', line=dict(color='#1a5276', width=2.5),
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
                mode='lines', line=dict(color='#1a5276', width=2.5),
                name=f'Projection (2x odds: {_user_dt_center:.0f}d, CI {rli_custom_dt_lo}\u2013{rli_custom_dt_hi}d)',
                hovertext=hover_proj, hoverinfo='text',
            ))
    elif rli_proj_basis == "Superexponential (logit)":
        d_start = int(days_used[0])
        d_end = (proj_end_date - base_date).days
        days_range = np.arange(d_start, d_end + 1, 1)
        logit_trend = _rli_se_A + _rli_se_K * 2 ** (days_range / rli_superexp_halflife)
        y_trend = _inv_logit(logit_trend) * 100
        dates_seg = [base_date + timedelta(days=int(d)) for d in days_range]
        hover_seg = [f"{dt.strftime('%b %d, %Y')}<br>Trend: {y:.2f}%" for dt, y in zip(dates_seg, y_trend)]
        fig.add_trace(go.Scatter(
            x=dates_seg, y=y_trend.tolist(),
            mode='lines', line=dict(color='#8e44ad', width=2.5),
            name=f'Superexp fit (2x odds: {rli_superexp_dt_fitted:.0f}d, HL={rli_superexp_halflife}d)',
            hovertext=hover_seg, hoverinfo='text',
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


    # --- Today vline ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

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
    st.plotly_chart(fig, use_container_width=True)
    if rli_is_backtesting and rli_backtest_results:
        _backtest_summary(rli_backtest_results)

    # ── Projections row ───────────────────────────────────────────────────
    rli_start_logit = rli_proj_start_logit
    rli_current_label = rli_current['name']

    eoy_targets = [
        ("Projected today", datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)),
        ("2026 Jun EOM", datetime(2026, 6, 30)),
        ("2026EOY", datetime(2026, 12, 31)),
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

    st.caption("RLI = Remote Labor Index (remotelabor.ai). Projections use logit-space fitting to keep scores bounded 0\u2013100%.")


# ── Dispatch ─────────────────────────────────────────────────────────────

if active_tab == "METR Horizon":
    render_metr()
elif active_tab == "Epoch ECI":
    render_eci()
elif active_tab == "Remote Labor Index":
    render_rli()
