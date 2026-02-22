"""
METR Frontier Projection: Interactive Plotly fan chart through EOY 2026 with text projections to 2029.
Run: streamlit run visualize_projection.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import yaml
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="METR Frontier Projection", layout="wide")

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
    if h < 8:
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


# ── Load frontier (before sidebar, so model names are available) ─────────

frontier_all = load_frontier(_mtime=_yaml_mtime())
gpt4o_idx = next(i for i, m in enumerate(frontier_all) if m['name'] == 'gpt_4o_inspect')
frontier_names = [pretty(m['name']) for m in frontier_all]


# ── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Projection")

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
                "DT CI low (days)", value=100,
                min_value=10, max_value=2000, step=5, key="custom_dt_lo")
            custom_dt_hi = custom_dt_hi.number_input(
                "DT CI high (days)", value=200,
                min_value=10, max_value=2000, step=5, key="custom_dt_hi")

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
    st.selectbox(
        "Project as of",
        frontier_names,
        index=frontier_names.index(proj_as_of_name),
        key='_proj_as_of',
        help="Backtest: project from an earlier model's vantage point.",
    )

# ── Reliability metric keys ──────────────────────────────────────────────
# Everything downstream uses these generic keys so the p50/p80 toggle works
_val_key = 'p80_min' if use_p80 else 'p50_min'
_lo_key = 'p80_lo' if use_p80 else 'p50_lo'
_hi_key = 'p80_hi' if use_p80 else 'p50_hi'
_reliability_label = "p80" if use_p80 else "p50"


# ── Build data arrays ────────────────────────────────────────────────────

# Truncate frontier for trend fitting
frontier_used = frontier_all[:proj_as_of_idx + 1]

# Plot frontier models (optionally filtered to GPT-4o onward)
if only_post_gpt4o:
    frontier_plot = list(frontier_all[gpt4o_idx:])
    plot_start_idx = gpt4o_idx  # offset into frontier_all for styling logic
else:
    frontier_plot = list(frontier_all)
    plot_start_idx = 0

base_date = frontier_all[0]['date']
days_all = np.array([(m['date'] - base_date).days for m in frontier_all], dtype=float)
# Always fit on p50 — p80 toggle only affects scatter display
log2_all = np.array([np.log2(m['p50_min']) for m in frontier_all])


# Arrays for the truncated (used) frontier — further filter to GPT-4o+ if toggled
_fit_start = gpt4o_idx if only_post_gpt4o else 0
_fit_end = proj_as_of_idx + 1
frontier_used = frontier_all[_fit_start:_fit_end]
days_used = days_all[_fit_start:_fit_end]
log2_used = log2_all[_fit_start:_fit_end]  # always p50 — for slope fitting
log2_disp_used = np.array([np.log2(m[_val_key]) for m in frontier_used])  # p50 or p80 — for intercept
n_used = len(frontier_used)


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
    # OLS doubling time from the fitted segment (used to center the DT CI on the trend)
    _cu_ols_dt = 1.0 / _cu_params[1] if _cu_params[1] > 0 else (custom_dt_lo + custom_dt_hi) / 2
    # Shift the user CI so its center aligns with the OLS DT, preserving the CI width ratio
    _cu_ci_center = np.sqrt(custom_dt_lo * custom_dt_hi)  # geometric mean of user CI
    _cu_dt_shift = _cu_ols_dt / _cu_ci_center
    _eff_dt_lo = custom_dt_lo * _cu_dt_shift
    _eff_dt_hi = custom_dt_hi * _cu_dt_shift
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
        # log(X) ~ Lognormal, centered on OLS-fitted (work in minutes for numerical safety)
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
    # Sample DT from user CI, re-centered on the fitted DT at current date
    n_superexp = 20000
    _se_ci_center = np.sqrt(superexp_dt_ci_lo * superexp_dt_ci_hi)
    _se_dt_shift = superexp_dt_fitted / _se_ci_center
    _se_eff_dt_lo = superexp_dt_ci_lo * _se_dt_shift
    _se_eff_dt_hi = superexp_dt_ci_hi * _se_dt_shift
    proj_dt = _lognormal_from_ci(_se_eff_dt_lo, _se_eff_dt_hi, n_superexp)
    # Position: lognormal noise centered on the fitted trend position
    _se_fitted_hrs = 2**_se_fitted_pos / 60
    _se_pos_sigma = (np.log(superexp_pos_hi) - np.log(superexp_pos_lo)) / (2 * 1.282)
    _se_pos_mu = np.log(_se_fitted_hrs)  # center on fitted trend, not raw model
    proj_start = np.log2(np.random.lognormal(_se_pos_mu, max(_se_pos_sigma, 0), n_superexp) * 60)


# ── Current SOTA (selected "as of" model) ────────────────────────────────

current = frontier_used[-1]
current_log2 = np.log2(current[_val_key])
current_hrs = current[_val_key] / 60


# ── Plotly chart ─────────────────────────────────────────────────────────

proj_end_date = datetime(2026, 12, 31)
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
        # Time when DT hits the floor: dt_0 * 2^(-t_cap/H) = floor → t_cap = H * log2(dt_0/floor)
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

# y-axis conversion: log2(minutes) → display value
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
                d0 = int(days_used[seg_idx[0]])
                d1 = (proj_end_date - base_date).days
                dates_seg, y_seg, hover_seg = _trend_hover(seg_params, d0, d1, base_date)
                fig.add_trace(go.Scatter(
                    x=dates_seg, y=y_seg,
                    mode='lines', line=dict(color='#2c3e50', width=2.5),
                    name=f'Segment {si+1} ({seg_dt:.0f}d doubling, CI {custom_dt_lo}\u2013{custom_dt_hi}d)',
                    hovertext=hover_seg, hoverinfo='text',
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
        # Single OLS through full used frontier, extended to EOY
        custom_params = _fit_slope_p50_intercept_display(days_used, log2_used, log2_disp_used)
        custom_ols_dt = 1.0 / custom_params[1] if custom_params[1] > 0 else float('inf')
        d0, d1 = int(days_used[0]), (proj_end_date - base_date).days
        dates_seg, y_seg, hover_seg = _trend_hover(custom_params, d0, d1, base_date)
        fig.add_trace(go.Scatter(
            x=dates_seg, y=y_seg,
            mode='lines', line=dict(color='#2c3e50', width=2.5),
            name=f'OLS trend ({custom_ols_dt:.0f}d doubling, CI {custom_dt_lo}\u2013{custom_dt_hi}d)',
            hovertext=hover_seg, hoverinfo='text',
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
today = datetime(2026, 2, 21)
fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
fig.add_annotation(
    x=today, y=1.0, yref='paper', text='Today', showarrow=False,
    font=dict(size=10, color='gray'), yanchor='top')

# --- Data points: distinguish used vs future ---
for idx_m, m in enumerate(frontier_plot):
    global_idx = idx_m + plot_start_idx  # index into frontier_all
    is_used = global_idx <= proj_as_of_idx
    is_selected = global_idx == proj_as_of_idx
    is_c46 = m['name'] == 'claude_opus_4_6_inspect'
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

st.plotly_chart(fig, use_container_width=True)

# ── Projections ───────────────────────────────────────────────────────────

start_hrs_samples = 2**proj_start / 60
med_dt = np.median(proj_dt)
p10_dt, p90_dt = np.percentile(proj_dt, [10, 90])
current_label = pretty(current['name'])

eoy_targets = [
    ("Projected today", datetime(2026, 2, 21)),
    ("2026 Jun EOM", datetime(2026, 6, 30)),
    ("Dec 2026", datetime(2026, 12, 31)),
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

# Current model + projected horizon row
_cur_lo = current.get(_lo_key)
_cur_hi = current.get(_hi_key)
_cur_ci = f"80% CI: {fmt_hrs(_cur_lo/60)} – {fmt_hrs(_cur_hi/60)}" if _cur_lo and _cur_hi else ""
n_proj_cols = len(eoy_targets)
cols = st.columns([1.2] + [1] * n_proj_cols)
with cols[0]:
    st.metric(label=f"{current_label} ({current['date'].strftime('%b %Y')})", value=fmt_hrs(current_hrs))
    if _cur_ci:
        st.caption(_cur_ci)
for col, (label, target_date) in zip(cols[1:], eoy_targets):
    elapsed = (target_date - current['date']).days
    proj_hrs = _proj_hrs_at(elapsed, start_hrs_samples, proj_dt, is_superexp, superexp_halflife, superexp_dt_floor if is_superexp else None)
    p10_h, p50_h, p90_h = np.percentile(proj_hrs, [10, 50, 90])
    with col:
        st.metric(label=label, value=fmt_hrs(p50_h))
        st.caption(f"80% CI: {fmt_hrs(p10_h)} – {fmt_hrs(p90_h)}")

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
                "80% CI": f"{early_date.strftime('%b %Y')} – {late_date.strftime('%b %Y')}",
            })
        st.table(arrival_rows)

st.caption("Time units are human work-time: 1d = 8h, 1w = 40h, 1mo = 176h, 1y = 2000h.")
