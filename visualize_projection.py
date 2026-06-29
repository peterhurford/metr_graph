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

PROJ_DISCLAIMER = " These are projections assuming current progress continues, they are not forecasts - forecasts would involve some chance of larger-scale trend changes."

# Tighten default Streamlit padding
st.markdown("""<style>
    .block-container { padding-top: 2rem !important; }
    [data-testid="stTable"] table { margin-top: 0 !important; margin-bottom: 0.5rem !important; }
</style>""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────

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
    return parent.number_input(label, key=key, value=st.session_state[key], **kwargs)


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
def load_eci_frontier(_mtime=None, country=None):
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

    # Filter to Claude 3 Opus era onward (Feb 2024+)
    _cutoff_date = datetime(2024, 2, 29)
    deduped = [m for m in deduped if m['date'] >= _cutoff_date]

    # Optional country filter
    if country:
        deduped = [m for m in deduped if m.get('country') == country]

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

    Used by the Compute vs Capabilities tab to regress ECI on log10(training
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


# ── Data center data (Epoch AI Frontier Data Centers) ────────────────────

import bisect

# 2-month training run (2 × 30-day months), used to turn a site's 8-bit OP/s
# throughput into total operations over a two-month run.
_DAYS_2MO = 2 * 30
_SECONDS_2MO = _DAYS_2MO * 24 * 3600
# A model ships ~1mo after its training run finishes (post-training, evals,
# safety), calibrated to observed train-finish → announce gaps. Used both to
# date model runs and to expand a buildout milestone into its DC-online /
# training / model-out timeline.
_CC_RUN_COMPLETION_LAG = timedelta(days=30)


def _dc_milestone_dates(record_date, shift_days=0):
    """Tooltip date lines for a buildout milestone. The plotted date is the site's
    DC-available date shifted forward by the chosen timing (`shift_days`), so
    reconstruct the base and expand into the three milestones the timing dropdown
    offers: DC online / training done / model out.
    """
    base = record_date - timedelta(days=shift_days)
    return (f"DC online ~{base:%b %d, %Y}<br>"
            f"Training done ~{(base + timedelta(days=_DAYS_2MO)):%b %d, %Y}<br>"
            f"Model out ~"
            f"{(base + timedelta(days=_DAYS_2MO) + _CC_RUN_COMPLETION_LAG):%b %d, %Y}")


# Fraction of peak throughput actually realized over a real training run.
_DC_UTILIZATION = 0.3

def _dc_mtime():
    p = os.path.join(os.path.dirname(__file__), 'data_center_timelines.csv')
    return os.path.getmtime(p)


def _dc_clean_owner(s):
    """Strip Epoch '#confident'/'#speculative' confidence tags from an owner field."""
    return (s or '').split('#')[0].strip()


@st.cache_data
def load_data_centers(_mtime=None):
    """Load Epoch's per-data-center capacity timelines plus an owner→company map.

    Returns a list of dicts: {name, company, points:[{date, status, h100,
    it_power, power, perf, cost}, ...]} with points sorted by date. Metric values
    are floats or None when missing.
    """
    base = os.path.dirname(__file__)

    # Operator (primary user) and owner maps from the data_centers metadata file.
    # We attribute each site to the AI lab operating it (its primary listed user),
    # falling back to the facility owner, then to the site-name token.
    user_by_dc = {}
    owner_by_dc = {}
    dc_meta_path = os.path.join(base, 'data_centers.csv')
    if os.path.exists(dc_meta_path):
        with open(dc_meta_path, 'r') as f:
            for r in csv.DictReader(f):
                name = (r.get('Name') or '').strip()
                if not name:
                    continue
                owner_by_dc[name] = _dc_clean_owner(r.get('Owner', ''))
                # Users may be a comma-separated list; take the primary (first).
                primary_user = (r.get('Users', '') or '').split(',')[0]
                user_by_dc[name] = _dc_clean_owner(primary_user)

    def company_for(dc_name):
        return (user_by_dc.get(dc_name, '')
                or owner_by_dc.get(dc_name, '')
                # Fallback for colocation sites with no listed user/owner:
                # use the first token of the site name (QTS, DayOne, EdgeCore…).
                or (dc_name.split()[0] if dc_name.split() else dc_name))

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
            series.setdefault(dname, []).append({
                'date': d,
                'status': r.get('Construction status', ''),
                'h100': _num(r, 'H100 equivalents'),
                'it_power': _num(r, 'IT power (MW)'),
                'power': _num(r, 'Power (MW)'),
                'perf': perf,
                'train_flop': train_flop,
                # How many GPT-5-scale (2e25 FLOP) / Mythos-scale (1e27 FLOP)
                # training runs the site's 2-month capacity could produce.
                'gpt5s': train_flop / 2e25 if train_flop is not None else None,
                'mythos': train_flop / 1e27 if train_flop is not None else None,
                'cost': _num(r, 'Total capital cost (2025 USD billions)'),
            })

    dcs = []
    for name, pts in series.items():
        pts.sort(key=lambda p: p['date'])
        dcs.append({'name': name, 'company': company_for(name), 'points': pts})
    dcs.sort(key=lambda dc: dc['name'])
    return dcs


# Metric options for the data-center tab: label → (point key, log-scale default, formatter kind)
_DC_METRICS = {
    "Compute (H100-equivalents)": {"key": "h100", "log": True, "kind": "h100"},
    "Power (MW)": {"key": "power", "log": False, "kind": "mw"},
    "IT power (MW)": {"key": "it_power", "log": False, "kind": "mw"},
    "Capital cost ($B)": {"key": "cost", "log": False, "kind": "cost"},
    "Performance (8-bit OP/s)": {"key": "perf", "log": True, "kind": "sci"},
    "2mo train FLOP": {"key": "train_flop", "log": True, "kind": "flop"},
    "Capacity (x GPT-5)": {"key": "gpt5s", "log": True, "kind": "count"},
    "Capacity (x Mythos)": {"key": "mythos", "log": True, "kind": "count"},
}

# Timing options for the 2mo-train-FLOP metric: label → days the DC-available
# date is shifted forward to date the chosen milestone.
#   • DC construction  — no shift (the site's availability date)
#   • Training done     — +2mo (a run started at availability finishes that later)
#   • Model release     — +2mo + ~1mo post-training/eval lag (matches the
#                         Compute vs Capabilities tab's model-release dating)
_DC_TRAIN_FLOP_TIMINGS = {
    "Data center construction": 0,
    "2mo training finished": _DAYS_2MO,
    "Model release": _DAYS_2MO + _CC_RUN_COMPLETION_LAG.days,
}

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
_DC_EXCLUDE_COMPANIES = {
    "QTS", "DayOne", "CoreWeave", "STACK", "Stream", "Vantage", "EdgeCore",
    "Oracle", "Microsoft",
}


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
        return f"{v:.2e} FLOP"
    if kind == "count":
        if v >= 100:
            return f"{v:,.0f}"
        if v >= 10:
            return f"{v:.1f}"
        return f"{v:.2f}"
    return f"{v:g}"


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
            out[dc['name']] = {'company': dc['company'], 'pts': pts}
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
        companies.setdefault(v['company'], []).append((name, v['pts']))
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


def _dc_color(company, idx):
    return _DC_COLORS.get(company, _DC_PALETTE[idx % len(_DC_PALETTE)])


# ── Load data (before sidebar, so model names are available) ─────────────

metr_all = load_metr_all(_mtime=_yaml_mtime())
frontier_all = [m for m in metr_all if m['is_sota']]
gpt4o_idx = next(i for i, m in enumerate(frontier_all) if m['name'] == 'gpt_4o_inspect')
frontier_names = [pretty(m['name']) for m in frontier_all]

eci_all = load_eci_frontier(_mtime=_eci_mtime())
eci_frontier_all = [m for m in eci_all if m['is_frontier']]
eci_frontier_names = [m['display_name'] for m in eci_frontier_all]

ecicn_all = load_eci_frontier(_mtime=_eci_mtime(), country='China')
ecicn_frontier_all = [m for m in ecicn_all if m['is_frontier']]
ecicn_frontier_names = [m['display_name'] for m in ecicn_frontier_all]

rli_all = load_rli_data()
rli_frontier_all = [m for m in rli_all if m['is_frontier']]
rli_frontier_names = [m['name'] for m in rli_frontier_all]

dc_all = load_data_centers(_mtime=_dc_mtime())


# ── Sidebar: tab selector ────────────────────────────────────────────────

_TAB_OPTIONS = ["METR Horizon", "Epoch ECI", "ECI Company Gap", "Remote Labor Index", "Prinz", "Revenue", "Employment", "Data Centers", "Compute vs Capabilities"]
_TAB_SLUG = {"metr": 0, "eci": 1, "rli": 2, "prinz": 3, "revenue": 4, "employment": 5, "ecigap": 6, "datacenters": 7, "computecap": 8}

# Read ?tab= from URL for deep-linking
_url_tab = st.query_params.get("tab", "").lower()
_default_tab_idx = _TAB_SLUG.get(_url_tab, 0)

with st.sidebar:
    active_tab = st.radio("Tab", _TAB_OPTIONS, index=_default_tab_idx, horizontal=True, key="_active_tab")
    st.markdown("---")

# Keep URL in sync with selected tab (omit when at default)
_SLUG_FOR_TAB = {"METR Horizon": "metr", "Epoch ECI": "eci", "Remote Labor Index": "rli", "Prinz": "prinz", "Revenue": "revenue", "Employment": "employment", "ECI Company Gap": "ecigap", "Data Centers": "datacenters", "Compute vs Capabilities": "computecap"}
_DEFAULT_TAB = _TAB_OPTIONS[0]
if active_tab == _DEFAULT_TAB:
    if "tab" in st.query_params:
        del st.query_params["tab"]
else:
    st.query_params["tab"] = _SLUG_FOR_TAB[active_tab]



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
        n_custom = 5000
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
        n_superexp = 5000
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

    st.caption("Fine print: Time units are human work-time: 1d = 8h, 1w = 40h, 1mo = 176h, 1y = 2000h." + PROJ_DISCLAIMER)


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
                    us_match_marker=None, us_match_label=None):
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

        n_eci = 5000
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

        n_eci = 5000
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

    # US-trend gap helpers (months China lags the US trend at a given score).
    # Only active on tabs that supply an overlay (US) frontier.
    _eci_gap_fn = None
    if overlay_frontier is not None and len(overlay_frontier) >= 2:
        # Same gap-in-time metric as the US-China section; share one definition.
        _us_fr_tuples = [(mm['date'], mm['eci_score'], '') for mm in overlay_frontier]

        def _eci_gap_fn(score, at_date):
            """Months `score` lags the US trend, evaluated at China date `at_date`."""
            return _eci_months_behind(_us_fr_tuples, score, at_date)

    def _eci_gap_hover(score, at_date):
        """'<br>Gap: X mo behind US' suffix, or '' when there's no overlay frontier."""
        if _eci_gap_fn is None:
            return ""
        g = _eci_gap_fn(score, at_date)
        if g >= 0:
            return f"<br>Gap: {g:.1f} mo behind US"
        return f"<br>Gap: {-g:.1f} mo ahead of US"

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

    # --- Today vline ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

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

    st.caption("Fine print: ECI = Epoch Capabilities Index. +Pts/Yr = ECI points gained per year." + PROJ_DISCLAIMER)



def render_eci():
    # Region toggle: the US frontier or China-only, sharing one tab.
    with st.sidebar:
        eci_region = st.radio(
            "Region", ["United States", "China"],
            key="eci_region", horizontal=True,
        )

    if eci_region == "China":
        _render_eci_china()
    else:
        _render_eci_tab(
            eci_all, eci_frontier_all, eci_frontier_names, "eci",
            "ECI Projection",
            [(155, "ECI 155", '#888888'), (160, "ECI 160", '#666666'),
             (165, "ECI 165", '#c0392b'), (170, "ECI 170", '#8e44ad')],
        )


def _render_eci_china():
    # Single milestone line at the current best US ECI (frontier max). The line
    # emanates rightward from a labeled dot at the model's actual release date.
    _us_best = max(eci_frontier_all, key=lambda m: m['eci_score'])
    _us_best_score = _us_best['eci_score']
    _us_best_label = f"US best {_us_best_score:.1f}"
    # US model closest to China's current best ECI — marks when the US passed
    # the level China sits at today (the lead time China currently lags).
    _cn_current = max(ecicn_frontier_all, key=lambda m: m['eci_score'])
    _us_match = min(eci_frontier_all, key=lambda m: abs(m['eci_score'] - _cn_current['eci_score']))
    _render_eci_tab(
        ecicn_all, ecicn_frontier_all, ecicn_frontier_names, "ecicn",
        "ECI China Projection",
        [(_us_best_score, _us_best_label, '#8e44ad')],
        overlay_frontier=eci_frontier_all,
        overlay_label="US trend",
        us_best_marker={
            'date': _us_best['date'],
            'score': _us_best_score,
            'name': _us_best['display_name'],
        },
        us_match_marker={
            'date': _us_match['date'],
            'score': _us_match['eci_score'],
            'name': _us_match['display_name'],
        },
        us_match_label=f"China best {_cn_current['eci_score']:.1f}",
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

        n_rli = 5000
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

        n_rli = 5000
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
    st.plotly_chart(fig, width="stretch")
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

    st.caption("Fine print: RLI = Remote Labor Index (remotelabor.ai). Projections use logit-space fitting to keep scores bounded 0\u2013100%." + PROJ_DISCLAIMER)


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
    ("2026-05-15", 40.0),
    ("2026-05-21", 45.0),
    ("2026-05-28", 47.0),
]


def _parse_revenue(data):
    dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in data]
    values = [v for _, v in data]
    return dates, values


def _rev_fit_and_project(dates, vals, n_recent, proj_end, n_samples=5000,
                          dt_lo_override=None, dt_hi_override=None):
    """Fit exponential (OLS in log-space) to last n_recent points,
    sample doubling-time fan chart, return (proj_dates, percentiles_dict, ols_dt, ols_dates, ols_vals)."""
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


# ── Prinzbench ───────────────────────────────────────────────────────────
#
# Scores are read off the published "prinzbench (full)" bar chart (0–99 scale),
# so they are approximate (±1). Release dates are sourced from
# epoch_capabilities_index.csv where the model appears.
#
# NOTE: "qwen-q3-max" has no exact match in the ECI data; it is mapped to
# qwen3-max (2025-09-24). If that date is wrong, fix it in this table.
#
# A trailing "*" on a name means the model was accessed in early testing,
# before public release (per the chart's footnote).
_PRINZ_RAW = [
    # name,                       score, date,         org,        country
    ("gpt-5.5-pro (extended)*",     82, "2026-04-23", "OpenAI",    "United States"),
    ("gpt-5.4-pro (extended)",      79, "2026-03-05", "OpenAI",    "United States"),
    ("gpt-5.5-thinking (hvy)*",     74, "2026-04-23", "OpenAI",    "United States"),
    ("gpt-5.4 (xhigh)",             69, "2026-03-05", "OpenAI",    "United States"),
    ("gpt-5.4-thinking (hvy)",      68, "2026-03-05", "OpenAI",    "United States"),
    ("gpt-5.4-thinking (ext)",      57, "2026-03-05", "OpenAI",    "United States"),
    ("gpt-5.2-thinking (ext)",      52, "2025-12-11", "OpenAI",    "United States"),
    ("gpt-5.3-codex-high",          52, "2026-02-05", "OpenAI",    "United States"),
    ("gemini-3.1-pro",              51, "2026-02-19", "Google",    "United States"),
    ("grok-4.20",                   43, "2026-02-17", "xAI",       "United States"),
    ("opus-4.8 (max)",              42, "2026-05-28", "Anthropic", "United States"),
    ("gemini-3-flash",              36, "2025-12-17", "Google",    "United States"),
    ("gemini-3-pro",                35, "2025-11-18", "Google",    "United States"),
    ("kimi-k2.5-thinking",          35, "2026-01-27", "Moonshot",  "China"),
    ("meta-muse-spark",             31, "2026-04-08", "Meta",      "United States"),
    ("glm-5.2",                     30, "2026-06-16", "Z.ai",      "China"),
    ("grok-4.1-thinking",           25, "2025-11-17", "xAI",       "United States"),
    ("qwen-q3-max",                 25, "2025-09-24", "Alibaba",   "China"),
    ("opus-4.7",                    25, "2026-04-16", "Anthropic", "United States"),
    ("grok-4",                      23, "2025-07-09", "xAI",       "United States"),
    ("deepseek-v4-pro",             23, "2026-04-24", "DeepSeek",  "China"),
    ("kimi-k2-thinking",            22, "2025-11-06", "Moonshot",  "China"),
    ("sonnet-4.5",                  19, "2025-09-29", "Anthropic", "United States"),
    ("opus-4.6",                    18, "2026-02-05", "Anthropic", "United States"),
    ("opus-4.5",                    13, "2025-11-24", "Anthropic", "United States"),
    ("sonnet 4.6",                   7, "2026-02-17", "Anthropic", "United States"),
]

_PRINZ_US_COLOR = "#4F8DFD"
_PRINZ_CN_COLOR = "#e74c3c"

# Per-company colors for the "Company" grouping mode.
_PRINZ_ORG_COLORS = {
    "OpenAI":    "#10a37f",
    "Anthropic": "#d97706",
    "Google":    "#4285F4",
    "xAI":       "#111111",
    "Meta":      "#9aa0c9",
    "Moonshot":  "#ff5a36",
    "Z.ai":      "#2da8e8",
    "Alibaba":   "#8e44ad",
    "DeepSeek":  "#1f3a93",
}
# Fallback palette for any org not in the map above.
_PRINZ_FALLBACK_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#0082c8", "#f58231",
    "#911eb4", "#46f0f0", "#f032e6", "#d2f53c", "#fabebe",
]


def _prinz_org_color(org, idx=0):
    return _PRINZ_ORG_COLORS.get(org, _PRINZ_FALLBACK_COLORS[idx % len(_PRINZ_FALLBACK_COLORS)])


def load_prinz_data():
    """Parse the prinzbench table into dicts sorted by date, with per-country
    running-max frontier flags."""
    models = []
    for name, score, date_str, org, country in _PRINZ_RAW:
        models.append({
            "name": name,
            "score": float(score),
            "date": datetime.strptime(date_str, "%Y-%m-%d"),
            "org": org,
            "country": country,
        })
    models.sort(key=lambda m: (m["date"], -m["score"]))
    # Per-country frontier: running max of score in date order.
    maxes = {}
    for m in models:
        c = m["country"]
        prev = maxes.get(c, -float("inf"))
        m["is_frontier"] = m["score"] > prev
        if m["is_frontier"]:
            maxes[c] = m["score"]
    return models


def _prinz_frontier_from(ms, end_date):
    """Running-max (date, score) step points for an arbitrary model list,
    extended flat to end_date. Returns the points and the set of frontier model
    ids (each model that set a new running max)."""
    pts = []
    fr_ids = set()
    cur = -float("inf")
    for m in sorted(ms, key=lambda x: x["date"]):
        if m["score"] > cur:
            cur = m["score"]
            pts.append((m["date"], cur))
            fr_ids.add(id(m))
    if pts and pts[-1][0] < end_date:
        pts.append((end_date, cur))
    return pts, fr_ids


def _prinz_frontier_points(models, country, end_date):
    """Running-max (date, score) step points for one country, extended to end_date."""
    pts, _ = _prinz_frontier_from(
        [m for m in models if m["country"] == country], end_date)
    return pts


def render_prinz():
    st.header("Prinzbench (full) — capability by release date")
    st.caption(
        "Scores read from the published prinzbench (full) chart (approximate, ±1). "
        "Source: [prinz-ai/prinzbench](https://github.com/prinz-ai/prinzbench). "
        "Release dates from Epoch's Capabilities Index data. "
        "A trailing * marks a model accessed in early testing, before public release."
    )

    models = load_prinz_data()

    with st.sidebar:
        st.header("Prinzbench")
        prinz_group = st.selectbox(
            "Group by",
            ["US vs China", "Company", "All models (one line)"],
            key="prinz_group", index=2,
        )
        prinz_region = st.radio(
            "Show", ["Both", "US only", "China only"],
            horizontal=True, key="prinz_region", index=0,
        )
        prinz_labels = st.radio(
            "Labels", ["Frontier only", "All", "None"],
            horizontal=True, key="prinz_labels", index=0,
        )
        prinz_frontier = st.checkbox("Show frontier lines", value=True, key="prinz_frontier")
        if st.button("Reset", key="_reset_prinz_btn"):
            for k in ("prinz_group", "prinz_region", "prinz_labels", "prinz_frontier"):
                st.session_state.pop(k, None)
            st.rerun()

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    end_date = max(m["date"] for m in models)
    right_edge = max(today, end_date) + timedelta(days=20)

    # Region filter applies to every grouping mode.
    shown = [m for m in models
             if prinz_region == "Both"
             or (prinz_region == "US only" and m["country"] == "United States")
             or (prinz_region == "China only" and m["country"] == "China")]

    # Build (label, color, model_list) groups for the selected grouping mode.
    if prinz_group == "Company":
        by_org = {}
        for m in shown:
            by_org.setdefault(m["org"], []).append(m)
        ordered = sorted(by_org.items(),
                         key=lambda kv: -max(x["score"] for x in kv[1]))
        groups = [(org, _prinz_org_color(org, i), ms)
                  for i, (org, ms) in enumerate(ordered)]
    elif prinz_group == "All models (one line)":
        groups = [("All models", _PRINZ_US_COLOR, shown)]
    else:  # US vs China
        groups = [(lbl, col, [m for m in shown if m["country"] == ctry])
                  for ctry, lbl, col in (
                      ("United States", "United States", _PRINZ_US_COLOR),
                      ("China", "China", _PRINZ_CN_COLOR))]
        groups = [g for g in groups if g[2]]

    fig = go.Figure()

    # ── Frontier step lines + scatter, one group at a time ───────────────
    for label, color, ms in groups:
        if not ms:
            continue
        ms_sorted = sorted(ms, key=lambda m: m["date"])
        fr_pts, fr_ids = _prinz_frontier_from(ms_sorted, right_edge)

        if prinz_frontier and len(fr_pts) >= 2:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in fr_pts], y=[p[1] for p in fr_pts],
                mode="lines", line=dict(color=color, width=2, shape="hv"),
                opacity=0.55, hoverinfo="skip",
                name=f"{label} frontier", showlegend=(prinz_group != "Company"),
            ))

        is_fr = [id(m) in fr_ids for m in ms_sorted]
        if prinz_labels == "All":
            text = [m["name"] for m in ms_sorted]
            mode = "markers+text"
        elif prinz_labels == "Frontier only":
            text = [m["name"] if f else "" for m, f in zip(ms_sorted, is_fr)]
            mode = "markers+text"
        else:
            text = None
            mode = "markers"
        fig.add_trace(go.Scatter(
            x=[m["date"] for m in ms_sorted], y=[m["score"] for m in ms_sorted],
            mode=mode,
            marker=dict(
                color=color,
                size=[13 if f else 9 for f in is_fr],
                symbol=["star" if f else "circle" for f in is_fr],
                line=dict(color="white", width=1),
            ),
            text=text, textposition="middle right",
            textfont=dict(size=9, color="#1a1a2e"),
            customdata=[(m["name"], m["org"], m["country"]) for m in ms_sorted],
            hovertemplate=(
                "%{customdata[0]}<br>%{customdata[1]} (%{customdata[2]})"
                "<br>prinzbench: %{y:.0f}<extra></extra>"
            ),
            name=label,
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=50, r=160, t=50, b=40),
        font=dict(color="#1a1a2e"),
        xaxis=dict(
            range=[min(m["date"] for m in models) - timedelta(days=20), right_edge],
            gridcolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="#1a1a2e"),
            zeroline=False,
        ),
        yaxis=dict(
            title="prinzbench (full) score",
            range=[0, max(m["score"] for m in models) * 1.1],
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
            tickfont=dict(color="#1a1a2e"),
            title_font=dict(color="#1a1a2e"),
        ),
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01,
                    bgcolor="rgba(255,255,255,0.95)",
                    font=dict(color="#1a1a2e")),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    st.plotly_chart(fig, width="stretch")

    # ── US vs China comparison ───────────────────────────────────────────
    us = [m for m in models if m["country"] == "United States"]
    cn = [m for m in models if m["country"] == "China"]
    if us and cn:
        us_best = max(us, key=lambda m: m["score"])
        cn_best = max(cn, key=lambda m: m["score"])
        gap = us_best["score"] - cn_best["score"]

        st.subheader("US vs China")
        c1, c2, c3 = st.columns(3)
        c1.metric("US best", f"{us_best['score']:.0f}", help=us_best["name"])
        c2.metric("China best", f"{cn_best['score']:.0f}", help=cn_best["name"])
        c3.metric("Gap (points)", f"{gap:+.0f}")

        # Lead time: when did the US frontier first reach China's current best?
        us_fr = _prinz_frontier_points(us, "United States", right_edge)
        us_cross = next((d for d, s in us_fr if s >= cn_best["score"]), None)
        if us_cross is not None:
            lead_days = (cn_best["date"] - us_cross).days
            if lead_days > 0:
                st.markdown(
                    f"The US frontier reached China's current best score "
                    f"(**{cn_best['score']:.0f}**, {pretty(cn_best['name'])}, "
                    f"{cn_best['date']:%b %Y}) back on **{us_cross:%b %Y}** — "
                    f"roughly **{lead_days/30.4:.1f} months** earlier. "
                    f"That is the lead China currently lags by on prinzbench."
                )
            else:
                st.markdown(
                    f"China's best ({cn_best['score']:.0f}) is at or above where the "
                    f"US frontier sat at the same time — no clear US lead on this metric."
                )
        st.caption(
            f"US: {len(us)} models · China: {len(cn)} models. "
            "Frontier = running maximum within each country over time."
        )


def _fmt_revenue(val):
    """Format revenue value in $B to human-readable string."""
    if val >= 1000:
        return f"${val/1000:.1f}T"
    if val >= 1:
        return f"${val:.1f}B"
    return f"${val*1000:.0f}M"


_REV_MILESTONES = [
    (50, "$50B"),
    (100, "$100B"),
    (200, "$200B"),
    (500, "$500B"),
    (1000, "$1T"),
]


def render_revenue():
    st.header("Revenue Projections (ARR)")

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
    _rev_backtesting = _rev_as_of_idx < len(_all_rev_dates) - 1
    oai_can_project = len(oai_vals_fit) >= 3
    ant_can_project = len(ant_vals_fit) >= 3

    # Clamp slider session state values if they exceed filtered data length
    if st.session_state.get('oai_n_recent', 0) > len(oai_vals_fit):
        st.session_state.pop('oai_n_recent', None)
    if st.session_state.get('ant_n_recent', 0) > len(ant_vals_fit):
        st.session_state.pop('ant_n_recent', None)

    with st.sidebar:
        st.header("Revenue Projection")
        rev_end_year = st.radio(
            "Project through", [2026, 2027, 2028, 2029],
            horizontal=True, key="rev_end_year", index=0)
        log_scale = st.checkbox("Log scale", value=True, key="rev_log_scale")
        rev_2025_only = st.checkbox("2025+ only", value=False, key="rev_2025_only")
        show_milestones = st.toggle("Milestones", value=True, key="rev_milestones")
        show_labels = st.toggle("Labels", value=True, key="rev_labels")

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
                oai_n_recent = st.slider("Fit to last N points", 3, len(oai_vals_fit),
                                          value=min(6, len(oai_vals_fit)), key="oai_n_recent")
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
                ant_n_recent = st.slider("Fit to last N points", 3, len(ant_vals_fit),
                                          value=min(6, len(ant_vals_fit)), key="ant_n_recent")
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

    proj_end = datetime(rev_end_year, 12, 31)

    # Filter display data for 2025+ if toggled (projections still use all data for fitting)
    _rev_cutoff = datetime(2025, 1, 1) if rev_2025_only else datetime(2000, 1, 1)
    _oai_display = [(d, v) for d, v in zip(openai_dates, openai_vals) if d >= _rev_cutoff]
    _ant_display = [(d, v) for d, v in zip(anthropic_dates, anthropic_vals) if d >= _rev_cutoff]
    oai_display_dates = [d for d, _ in _oai_display]
    oai_display_vals = [v for _, v in _oai_display]
    ant_display_dates = [d for d, _ in _ant_display]
    ant_display_vals = [v for _, v in _ant_display]
    x_start = min(oai_display_dates[0], ant_display_dates[0]) - timedelta(days=30)

    oai_proj_dates = oai_pcts = oai_dt = oai_ols_dates = oai_ols_vals = oai_traj = None
    ant_proj_dates = ant_pcts = ant_dt = ant_ols_dates = ant_ols_vals = ant_traj = None

    if oai_can_project:
        oai_proj_dates, oai_pcts, oai_dt, oai_dt_lo_eff, oai_dt_hi_eff, oai_ols_dates, oai_ols_vals, oai_traj = \
            _rev_fit_and_project(oai_dates_fit, oai_vals_fit, oai_n_recent, proj_end,
                                  dt_lo_override=oai_dt_lo, dt_hi_override=oai_dt_hi)

    if ant_can_project:
        ant_proj_dates, ant_pcts, ant_dt, ant_dt_lo_eff, ant_dt_hi_eff, ant_ols_dates, ant_ols_vals, ant_traj = \
            _rev_fit_and_project(ant_dates_fit, ant_vals_fit, ant_n_recent, proj_end,
                                  dt_lo_override=ant_dt_lo, dt_hi_override=ant_dt_hi)

    # Backtest stats
    oai_bt_results = []
    ant_bt_results = []
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

    fig = go.Figure()

    # --- Fan bands ---
    _bt_results_map = {"OpenAI": oai_bt_results, "Anthropic": ant_bt_results}
    _can_project_map = {"OpenAI": oai_can_project, "Anthropic": ant_can_project}

    _rev_companies = []
    for _name, _proj_dates, _pcts, _color, _disp_dates, _disp_vals, \
            _ols_dates_raw, _ols_vals_raw, _dt in [
        ("OpenAI", oai_proj_dates, oai_pcts, '#10a37f', oai_display_dates, oai_display_vals,
         oai_ols_dates, oai_ols_vals, oai_dt),
        ("Anthropic", ant_proj_dates, ant_pcts, '#d4a574', ant_display_dates, ant_display_vals,
         ant_ols_dates, ant_ols_vals, ant_dt),
    ]:
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
        for bt_results, bt_color in [(oai_bt_results, '#10a37f'), (ant_bt_results, '#d4a574')]:
            if len(bt_results) >= 2:
                fig.add_trace(go.Scatter(
                    x=[r['date'] for r in bt_results],
                    y=[r['value'] for r in bt_results],
                    mode='lines',
                    line=dict(color=bt_color, width=2, dash='dash'),
                    hoverinfo='skip', showlegend=False,
                ))

    # --- Today vline ---
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

    # --- Layout ---
    yaxis_type = "log" if log_scale else "linear"
    all_display_vals = oai_display_vals + ant_display_vals
    y_min_data = min(all_display_vals)
    # Use 90th pctile (not 95th) to set range — 95th can be absurdly large
    _y_max_parts = []
    if oai_pcts is not None:
        _y_max_parts.append(oai_pcts[90][-1])
    if ant_pcts is not None:
        _y_max_parts.append(ant_pcts[90][-1])
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
    if show_milestones:
        with st.expander("Milestone details"):
            for name, p_dates, traj in [("OpenAI", oai_proj_dates, oai_traj),
                                          ("Anthropic", ant_proj_dates, ant_traj)]:
                if p_dates is None or traj is None:
                    continue
                arrival_rows = []
                for val, label in _REV_MILESTONES:
                    if val <= max(all_display_vals):
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

    st.caption("Fine print: Revenue figures are approximate ARR (annualized run rate) compiled from public reports and media sources. Anthropic Dec 2025 figure averaged from $8-10B range reported. May 2026 figures from [The Information](https://www.theinformation.com/articles/anthropic-openais-share-ai-startup-revenues-rises-89) and [The Information](https://www.theinformation.com/articles/openai-held-1-billion-revenue-lead-anthropic-first-quarter)." + PROJ_DISCLAIMER)


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

    n_emp = 5000
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

    # Today vline
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    fig.add_vline(x=today, line=dict(color='gray', width=1, dash='dash'), opacity=0.5)
    fig.add_annotation(
        x=today, y=1.0, yref='paper', text='Today', showarrow=False,
        font=dict(size=10, color='gray'), yanchor='top')

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

    st.caption("Fine print: This is a simple economic displacement model. "
               "RLI data from remotelabor.ai. "
               "The model assumes AI automation directly displaces remote/digital work proportional to a lagged RLI score, "
               "with partial recovery from Jevons paradox and worker reallocation. "
               "It does not model labor market dynamics, wage effects, new job creation, or policy responses."
               + PROJ_DISCLAIMER)


# ── ECI Company Gap ────────────────────────────────────────────────────

# Map raw CSV org names → display names
_ECG_ORG_MAP = {
    "OpenAI": "OpenAI",
    "Anthropic": "Anthropic",
    "Google DeepMind": "Google",
    "Meta AI": "Meta",
    "xAI": "xAI",
    "Mistral AI": "Mistral",
    "DeepSeek": "DeepSeek",
    "DeepSeek,Peking University": "DeepSeek",
    "Moonshot": "Moonshot",
    "Alibaba": "Alibaba",
    "Z.ai (Zhipu AI)": "Zhipu AI",
    "Z.ai (Zhipu AI),Tsinghua University": "Zhipu AI",
}

_ECG_COLORS = {
    "OpenAI": "#10a37f",
    "Anthropic": "#d4a574",
    "Google": "#4285F4",
    "Meta": "#0668E1",
    "xAI": "#555555",
    "Mistral": "#FF7000",
    "DeepSeek": "#1e90ff",
    "Moonshot": "#9B59B6",
    "Alibaba": "#E74C3C",
    "Zhipu AI": "#2ECC71",
}

_ECG_DASH = {
    "OpenAI": "solid", "Anthropic": "solid", "Google": "solid",
    "Meta": "solid", "xAI": "solid", "Mistral": "dash",
    "DeepSeek": "dot", "Moonshot": "dashdot", "Alibaba": "dot", "Zhipu AI": "dashdot",
}

_ECG_COUNTRY = {
    "OpenAI": "US", "Anthropic": "US", "Google": "US", "Meta": "US", "xAI": "US",
    "Mistral": "FR",
    "DeepSeek": "CN", "Moonshot": "CN", "Alibaba": "CN", "Zhipu AI": "CN",
}

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
        display = _ECG_ORG_MAP.get(org_raw)
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



    st.caption("Fine print: 'Gap at Release' interpolates when the overall ECI frontier "
               "first reached each model's score level at the time of release. "
               "'Gap Now' is how many months ago frontier first reached the org's current best score — "
               "zero iff they still tie or exceed the current frontier. "
               "'ECI Gap Now' is the ECI-point gap to the current frontier. "
               "Only each org's running-max (best) models are shown. "
               "ECI data from Epoch AI.")


# ── Data Centers ───────────────────────────────────────────────────────────

_DC_RESET_KEYS = ["dc_metric", "dc_log", "dc_future", "dc_timing"]
_DC_DEFAULTS = {
    "dc_metric": "Compute (H100-equivalents)",
    "dc_log": True,
    "dc_future": True,
    "dc_timing": "Data center construction",
}

# Compute vs Capabilities tab
_CC_RESET_KEYS = ["cc_future"]
_CC_DEFAULTS = {"cc_future": True}

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


def _dc_layout(log_scale, y_title, x_start, x_end, y_range=None,
               height=440, show_legend=False):
    return dict(
        height=height,
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=60, r=30, t=10, b=40),
        font=dict(color='#222222'),
        showlegend=show_legend,
        legend=dict(font=dict(size=11, color='#222222'), groupclick='togglegroup'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.12)', range=[x_start, x_end],
                   tickfont=dict(color='#222222'), title_font=dict(color='#222222')),
        yaxis=dict(title_text=y_title,
                   type='log' if log_scale else 'linear',
                   range=y_range,
                   gridcolor='rgba(0,0,0,0.12)',
                   tickfont=dict(color='#222222'), title_font=dict(color='#222222')),
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
        timing_label = st.selectbox(
            "Date points at", list(_DC_TRAIN_FLOP_TIMINGS), key="dc_timing")
        if st.button("Reset", key="dc_reset"):
            for k in _DC_RESET_KEYS:
                st.session_state.pop(k, None)
            # Re-seed defaults so they win over URL re-hydration on the rerun.
            st.session_state.update(_DC_DEFAULTS)
            st.rerun()

    key = cfg["key"]
    kind = cfg["kind"]
    # Cap projected buildout at end of 2028 when showing the future.
    cap_date = datetime(2028, 12, 31) if include_future else _today
    series = _dc_series_for_metric(dc_all, key, cap_date=cap_date)
    # Drop colocation / neutral-host providers (not AI labs).
    series = {n: v for n, v in series.items()
              if v['company'] not in _DC_EXCLUDE_COMPANIES}

    # Shift every data point forward from the site's availability date to the
    # chosen milestone (DC construction / training done / model release). DC
    # construction means no shift.
    shift_days = _DC_TRAIN_FLOP_TIMINGS[timing_label]
    if shift_days:
        shift = timedelta(days=shift_days)
        series = {n: {'company': v['company'],
                      'pts': [(d + shift, val) for d, val in v['pts']]}
                  for n, v in series.items()}
        cap_date = cap_date + shift

    # ── Header ──
    st.header("Frontier Data Centers Over Time")

    if key == 'train_flop':
        st.caption(
            f"Methodology: *2mo train FLOP* = each site's peak performance "
            f"(8-bit OP/s) × a {_DAYS_2MO}-day ({_DAYS_2MO // 30}-month) training "
            f"run × {_DC_UTILIZATION:.0%} realized utilization. Figures are "
            "order-of-magnitude estimates, not vendor-reported numbers.")
    elif key in ('gpt5s', 'mythos'):
        scale = "2e25 FLOP (GPT-5 scale)" if key == 'gpt5s' else "1e27 FLOP (Mythos scale)"
        st.caption(
            f"Methodology: how many {scale} training runs each site's 2mo train "
            f"FLOP could produce — i.e. *2mo train FLOP* ÷ {scale.split(' ')[0]}. "
            f"2mo train FLOP = peak performance (8-bit OP/s) × a {_DAYS_2MO}-day "
            f"run × {_DC_UTILIZATION:.0%} realized utilization. Order-of-magnitude "
            "estimates, not vendor-reported numbers.")

    if not series:
        st.warning("No data available for this metric.")
        return

    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Largest single data center over time
    # ══════════════════════════════════════════════════════════════════════
    env = _dc_envelope(series)
    st.subheader("Largest single data center")

    if env:
        cd, cv, cn, cco = env[-1]

    env_dates = [e[0] for e in env]
    env_vals = [e[1] for e in env]
    end_x = cap_date if cap_date is not None else (
        env_dates[-1] if env_dates else _today)
    # Focus the view on the AI buildout era; earlier milestones (land clearing
    # back to 2018) are off-screen but still carried forward into 2024.
    x_start = datetime(2024, 1, 1)
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
            hovertext=[f"{p[2]} ({p[3]}){' — planned' if projected else ''}<br>"
                       f"{_dc_fmt_value(p[1], kind)}<br>{_dc_milestone_dates(p[0], shift_days)}"
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
            hovertext=[f"{p[2]} ({p[3]}) — scale-up"
                       f"{' — planned' if p[0] > _today else ''}<br>"
                       f"{_dc_fmt_value(p[1], kind)}<br>"
                       f"{_dc_milestone_dates(p[0], shift_days)}"
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
                       f"{cn} ({cco})<br>{_dc_fmt_value(cv, kind)}<br>"
                       f"{_dc_milestone_dates(cd, shift_days)}"],
            hoverinfo='text', showlegend=False,
        ))
    fig1.update_layout(**_dc_layout(
        log_scale, metric_label, x_start, x_end,
        y_range=_dc_yrange(_dc_visible_vals(env, x_start), log_scale)))
    st.plotly_chart(fig1, use_container_width=True)

    comp = _dc_company_series(series)
    peaks = {co: max(v for _, v, _ in steps)
             for co, steps in comp.items() if steps}
    ranked = sorted(peaks, key=lambda c: peaks[c], reverse=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: Largest current data center by company (snapshot bar chart)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Largest current data center by company")
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
        y=[s[0] for s in snap],
        orientation='h',
        marker=dict(color=[_dc_color(s[0], i) for i, s in enumerate(snap)]),
        text=[_dc_fmt_value(s[1], kind) for s in snap],
        textposition='outside',
        hovertext=[f"{s[0]} — {s[2]}<br>{_dc_fmt_value(s[1], kind)}" for s in snap],
        hoverinfo='text',
    ))
    fig_snap.update_layout(
        height=max(300, 38 * len(snap) + 80),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=120, r=70, t=10, b=40),
        font=dict(color='#222222'), showlegend=False,
        xaxis=dict(title_text=f"Current {metric_label}",
                   type='log' if log_scale else 'linear',
                   gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222222'),
                   title_font=dict(color='#222222')),
        yaxis=dict(autorange='reversed', tickfont=dict(color='#222222')),
    )
    st.plotly_chart(fig_snap, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: Largest data center by company over time
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Largest data center by company over time")
    st.caption("Each line is a company's biggest single site at that time "
               "(not the sum of all its sites). Solid = actual; dashed = planned "
               "/ under construction.")

    fig2 = go.Figure()
    if include_future:
        _dc_add_projection_band(fig2, _today, x_end)
    for i, co in enumerate(ranked):
        color = _dc_color(co, i)
        steps = comp[co]
        (a_x, a_y), (p_x, p_y) = _dc_split_at(steps, _today, end_x)
        # Solid actual / dashed projected lines (hover handled by the dots).
        fig2.add_trace(go.Scatter(
            x=a_x, y=a_y, mode='lines',
            line=dict(color=color, width=2.5, shape='hv'),
            name=co, legendgroup=co, hoverinfo='skip',
        ))
        if p_x is not None:
            fig2.add_trace(go.Scatter(
                x=p_x, y=p_y, mode='lines',
                line=dict(color=color, width=2.5, shape='hv', dash='dash'),
                name=co, legendgroup=co, showlegend=False, hoverinfo='skip',
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
                name=co, legendgroup=co, showlegend=False,
                hovertext=[
                    f"{co} — {s[2]} ({label})"
                    f"{' (planned)' if s[0] > _today else ''}<br>"
                    f"{_dc_fmt_value(s[1], kind)}<br>{_dc_milestone_dates(s[0], shift_days)}"
                    for s in pts],
                hoverinfo='text',
            ))

        _dot_trace(new_dots, hollow=False, label="new leading site")
        _dot_trace(scale_dots, hollow=True, label="scale-up")
    comp_vals = [v for steps in comp.values()
                 for v in _dc_visible_vals(steps, x_start)]
    fig2.update_layout(**_dc_layout(log_scale, metric_label, x_start, x_end,
                                    y_range=_dc_yrange(comp_vals, log_scale),
                                    height=500, show_legend=True))
    st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: Quarterly capacity-by-company table
    # ══════════════════════════════════════════════════════════════════════
    _table_cos = ["OpenAI", "Anthropic", "Google", "Meta", "SpaceXAI", "Alibaba"]
    # Epoch's operator names don't always match the display label.
    _co_aliases = {"Google": ("Google", "Google DeepMind")}
    _q_ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}

    def _steps_for(co):
        for k in _co_aliases.get(co, (co,)):
            if k in comp:
                return comp[k]
        return []

    def _co_val_at(steps, t):
        """Forward-filled company value at date t (latest step on/before t)."""
        cur = None
        for d, v, _n in steps:
            if d <= t:
                cur = v
            else:
                break
        return cur

    _col_steps = {co: _steps_for(co) for co in _table_cos}
    md = ["| Quarter | " + " | ".join(_table_cos) + " |",
          "|" + "---|" * (len(_table_cos) + 1)]
    for yr in range(2024, 2030):
        for q in range(1, 5):
            mo, day = _q_ends[q]
            qd = datetime(yr, mo, day)
            cells = [_dc_fmt_value(_co_val_at(_col_steps[co], qd), kind)
                     for co in _table_cos]
            md.append(f"| {yr}Q{q} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(md))

    st.caption(
        f"Each cell is the company's largest single data center by **{metric_label}** "
        "as of the end of that quarter, forward-filled between Epoch AI's reported "
        "milestones. "
        f"{'Includes planned / under-construction buildout dated past today.' if include_future else 'Planned future buildout is excluded (toggle it in the sidebar); quarters past today repeat the latest known value.'} "
        "Data: Epoch AI, ‘Frontier Data Centers’ (epoch.ai/data/data-centers), CC-BY 4.0.")


# ══════════════════════════════════════════════════════════════════════════
# Compute vs Capabilities — does data-center FLOP predict ECI?
# ══════════════════════════════════════════════════════════════════════════

def _cc_trainflop_frontier(dcs, cap_date, with_names=False):
    """Running-max frontier of 2mo-train-FLOP across AI-lab sites.

    Returns [(date, flop), …] sorted by date, with every point shifted forward
    by the 2-month training lead time (a run on a site available at D only
    finishes at D+2mo), matching the Data Centers tab. Colocation/neutral-host
    providers are excluded. Monotonic non-decreasing.
    """
    series = _dc_series_for_metric(dcs, 'train_flop', cap_date=cap_date)
    series = {n: v for n, v in series.items()
              if v['company'] not in _DC_EXCLUDE_COMPANIES}
    shift = timedelta(days=_DAYS_2MO)
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
                     share_lo, share_mid, share_hi):
    """Quarterly frontier-ECI projection to end-2029.

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
    horizon = datetime(2029, 12, 31)
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

    N = 20000
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
    st.caption(
        "The median rise is split into its two engines: the **blue band** is the "
        f"algorithmic-efficiency contribution (~{algo_end:.0f} ECI by end-2029, a "
        "straight climb at a constant per-year rate) and the **red band** on top is "
        f"the physical-compute contribution (~{phys_end:.0f} ECI), which curves as "
        "it rides the projected compute frontier and flattens as the buildout "
        "matures. Compute is the thinner, decelerating slice; algorithms do most of "
        "the lifting at these horizons.")

    # Year-end milestone cards, each with its estimated METR p50 horizon.
    cols = st.columns(4)
    for col, yr in zip(cols, (2026, 2027, 2028, 2029)):
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

    st.caption(
        "Method: frontier ECI = today's anchor + physical-compute growth (the "
        f"physical share of ~{obs_slope:.0f} ECI/yr, scaled to the projected "
        "compute-frontier path so it decelerates with the buildout) + a constant "
        "algorithmic-efficiency term. Bands sample the contested physical/algo "
        f"mix ({s_lo*100:.0f}–{s_hi*100:.0f}%), compute delivery (×0.5–1.15 of "
        "plan), and overall pace (±12%). METR horizons come from the org-neutral "
        "ECI→METR fits `p50_min = 2^(0.24·ECI − 28.68)` and `p80_min = "
        "2^(0.23·ECI − 29.95)` (the METR ranges shown are driven by the ECI 80% "
        "band, not the fits' own ±bands). "
        "This extrapolates the historical linear ECI trend; it assumes no "
        "paradigm shift, no hard compute wall, and that ECI stays a meaningful "
        "scale at these levels — order-of-magnitude, not a promise.")


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
# pipeline (~1.9×/yr). Those segment slopes are read live from the DC data, so no
# US constant lives here.
#
# China is a JUDGMENT call: we have no Epoch buildout data for China (its DC
# tracking is ~all US sites). Export controls cap leading-edge chip supply (no
# H100/B200; domestic Ascend/SMIC yield-limited), so its buildout is unlikely to
# keep US pace. Low end = controls bite / stockpiles deplete; high end = China
# sustains its most recent disclosed leg (~2×/yr). This is the gap we're trying to
# bound, not measure.
_CC_CN_COMPUTE_LO = 0.15   # ~1.4×/yr — export controls bite
_CC_CN_COMPUTE_HI = 0.30   # ~2×/yr — China's recent disclosed pace holds

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


def _cc_us_vs_china(cc_rows, today):
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

    horizon = datetime(2029, 12, 31)

    # ── Compute: largest *actual* runs (grounded, Epoch per-model) vs cluster
    # *capacity* (US measured from buildout; China estimated from its chips). ──
    # Actual-run frontiers are us_cf / cn_cf (Epoch's per-model estimates). The
    # headline gap is run-to-run.
    us_run_lf, cn_run_lf = us_cf[-1][1], cn_cf[-1][1]
    run_gap_oom = us_run_lf - cn_run_lf

    # US capacity = Epoch's data-center buildout (largest cluster's 3-mo training
    # capacity); its recent-built vs planned segments give the forward range.
    dc_fr = _cc_trainflop_frontier(dc_all, horizon, with_names=True)
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
    # headroom (no Epoch buildout data; compute-constrained). Anchored at the last
    # run so the fan connects to the data: lower edge = the run, upper = +headroom.
    g_cn_lo, g_cn_hi = _CC_CN_COMPUTE_LO, _CC_CN_COMPUTE_HI
    cn_cap_d = cn_cf[-1][0] - _CC_RUN_COMPLETION_LAG
    cn_cap_apex_lo, cn_cap_apex_hi = cn_run_lf, cn_run_lf + _CC_CN_CAPACITY_HEADROOM_OOM
    d_yrs = (today - cn_cap_d).days / 365.25
    cn_cap_lo_lf = cn_cap_apex_lo + g_cn_lo * d_yrs     # capacity range at today
    cn_cap_hi_lf = cn_cap_apex_hi + g_cn_hi * d_yrs
    cn_cap_lf = 0.5 * (cn_cap_lo_lf + cn_cap_hi_lf)
    cap_gap_oom = us_cap_lf - cn_cap_lf

    # ECI projection: derived from compute (Chart A growth) + shared algorithmic
    # progress. a_partial = ECI per ×10 compute; b_algo = shared ECI/yr at fixed
    # compute (methods diffuse). Each country's ECI slope = a_partial·g + b_algo.
    a_partial, b_algo = _cc_pooled_decomp(cc_rows)
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
    c3.metric("ECI gap end-2029 (compute + algo)", f"~{gap_end:.0f} pts",
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
            text=[f"<b>{n}</b><br>{10.0 ** lf:.1e} FLOP &nbsp; ECI {s:.0f}<br>"
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
        text=[f"<b>{n}</b><br>{10.0 ** lf:.1e} FLOP (2-mo run)<br>"
              f"{_dc_milestone_dates(sd, _DAYS_2MO)}"
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
        yaxis=dict(title_text="Training compute (FLOP)", type='log',
                   gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222'),
                   title_font=dict(color='#222')))
    st.plotly_chart(figc, use_container_width=True)
    st.caption(
        f"**Solid = grounded**: largest *actual* training runs Epoch estimates per "
        f"model — ~{10 ** us_run_lf:.0e} (US) vs ~{10 ** cn_run_lf:.0e} (China), a "
        f"**~{10 ** run_gap_oom:.0f}× ({run_gap_oom:.1f} OOM)** gap. Recent US "
        f"frontier models use *less* (GPT-5 ~7e25) — efficiency, not bigger runs. "
        f"**Dashed/shaded = capacity** — the largest *single cluster's* 2-month "
        f"run: US ~{10 ** us_cap_lf:.0e} today rising through *announced* "
        f"megaclusters (Stargate, Hyperion…) to ~{10 ** us_cap_end_lf:.0e} by 2029 "
        f"(Epoch buildout), China ~{10 ** cn_cap_lo_lf:.0e}–{10 ** cn_cap_hi_lf:.0e} "
        f"(estimated, no Epoch data). "
        f"China's *total* chips — mostly smuggled NVIDIA + domestic Ascend — are "
        f"plentiful, but concentrating them into one coherent run is the bottleneck "
        f"(dispersed chips, export-controlled networking), so single-cluster "
        f"capacity stays near its biggest known run (~{10 ** cn_run_lf:.0e}). "
        f"Capacity grows {10 ** g_us_lo:.1f}–{10 ** g_us_hi:.1f}×/yr (US, measured) "
        f"vs ~{10 ** g_cn_lo:.1f}–{10 ** g_cn_hi:.1f}×/yr (China), so the gap "
        f"widens only slowly. *Run points are dated at estimated training "
        "completion (release − ~1mo) to align with the +2mo capacity line.*")

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
    st.caption(
        f"ECI projected as **~{a_partial:.0f} pts per ×10 compute** (pooled US+China "
        f"exchange rate) riding each country's Chart-A compute growth, **plus a "
        f"shared ~{b_algo:.0f} pts/yr algorithmic term** (methods diffuse, so it's "
        f"the same for both). Band = each country's compute-growth range; the "
        f"divergence is purely the compute gap, so the ECI gap widens slowly — to "
        f"~{gap_end:.0f} pts (~{mo_end:.0f} mo) by end-2029. The **dotted line** "
        "under each country is the algorithmic-only climb (shared rate, no compute); "
        "the **shaded gap** up to the dashed line is that country's compute "
        "contribution — visibly wider for the US than for China.")

    st.caption(
        "Caveats: a_partial and b_algo come from a pooled OLS where compute and "
        "time are collinear, so the compute-vs-algorithm split is approximate; US "
        "labs under-disclose training FLOP, so the compute gap is if anything "
        "understated; and no forward data-center data exists for China (Epoch's "
        "buildout tracking is almost entirely US sites). US = 'United States of "
        "America', China = 'China' in Epoch's country tags; multi-country and "
        "untagged models excluded. Order-of-magnitude, not forecasts.")

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

        # Backdate the scenario to the start of 2025: China's distillation edge has
        # been operating through the whole post-DeepSeek-V3 era, so the two algo-rate
        # trajectories diverge from a Jan-2025 anchor rather than from today — the
        # premium accumulates across the full period it actually applied to. The
        # China-own-rate line then doubles as a backtest: fit to the real data, it
        # should track China's actual frontier, while the US-rate line is the
        # counterfactual "where China would be on the leader's algo rate alone."
        anchor_date = datetime(2025, 1, 1)

        def _fr_at(fr, d):
            vals = [s for dd, s, n in fr if dd <= d]
            return max(vals) if vals else fr[0][1]

        cn_anchor = _fr_at(cn_fr, anchor_date)
        us_anchor = _fr_at(us_fr, anchor_date)
        bd_dates = [anchor_date] + _cc_quarter_ends(anchor_date, horizon)
        dt_bd = np.array([(d - anchor_date).days / 365.25 for d in bd_dates])

        # Both trajectories capped at the US frontier — distillation is a *catch-up*
        # mechanism (it can't exceed its teacher), so the high-algo line converges
        # toward the US line, never past it.
        us_ceiling = us_anchor + us_eci_smid * dt_bd
        cn_traj_us = cn_anchor + slope_usalgo * dt_bd
        cn_traj_cn = np.minimum(cn_anchor + slope_cnalgo * dt_bd, us_ceiling)

        us_end = float(us_ceiling[-1])
        cn_end_us = float(cn_traj_us[-1])
        cn_end_cn = float(cn_traj_cn[-1])
        mo_us = (us_end - cn_end_us) / us_eci_smid * 12 if us_eci_smid > 0 else float('nan')
        mo_cn = (us_end - cn_end_cn) / us_eci_smid * 12 if us_eci_smid > 0 else float('nan')

        st.markdown(
            "| Scenario | China algo rate | China ECI/yr (algo+compute) | "
            "China ECI end-2029 | Behind US |\n"
            "|---|---|---|---|---|\n"
            f"| **US algo growth** (shared) | {us_algo:.1f} pts/yr | "
            f"{slope_usalgo:.1f} pts/yr | ~{cn_end_us:.0f} | ~{mo_us:.0f} mo |\n"
            f"| **China's own algo growth** (distillation) | {cn_algo:.1f} pts/yr | "
            f"{slope_cnalgo:.1f} pts/yr | ~{cn_end_cn:.0f} | ~{mo_cn:.0f} mo |")

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
            name=f'China · US algo rate ({us_algo:.1f})',
            hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra>US algo</extra>'))
        figd.add_trace(go.Scatter(
            x=bd_dates, y=list(cn_traj_cn), mode='lines',
            line=dict(color='#7F1010', width=2.5),
            name=f"China · own algo rate ({cn_algo:.1f})",
            hovertemplate='%{x|%b %Y}<br>ECI %{y:.0f}<extra>China algo</extra>'))
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
        st.caption(
            f"Backdated to **Jan 2025** (China frontier ≈{cn_anchor:.0f} ECI then), "
            "so the distillation edge accumulates over the whole period it's applied. "
            f"Both China lines share the same compute term (~{compute_term_cn:.1f} "
            f"ECI/yr, from {a_partial:.0f} pts/×10 compute × China's "
            f"{10 ** g_cn_mid:.1f}×/yr capacity growth); they differ only in the "
            "algorithmic term. The **solid dark line** uses China's own (higher) "
            f"empirically-measured rate ({cn_algo:.1f} vs {us_algo:.1f} ECI/yr, "
            f"a +{premium:.1f}/yr ~+{premium_pct:.0f}% edge) and is **capped at the "
            "US frontier** — distillation closes the gap faster but can't push a "
            "follower past the model it learns from; it tracks China's actual points, "
            "while the dashed **US-rate** line is the counterfactual where China would "
            "sit on the leader's algo rate alone. So the edge buys China earlier "
            "*parity*, not a lead. "
            f"Iso-compute fits use n={n_us_iso} US / n={n_cn_iso} China models within "
            "±0.4 dex of each country's median compute; with so few same-budget US "
            "models and a flat China compute axis, treat the edge as suggestive, not "
            "established.")


def render_compute_capabilities():
    _today = datetime.now()

    with st.sidebar:
        st.header("Compute vs Capabilities")
        include_future = st.checkbox("Include planned future buildout",
                                     value=True, key="cc_future")
        if st.button("Reset", key="cc_reset"):
            for k in _CC_RESET_KEYS:
                st.session_state.pop(k, None)
            st.session_state.update(_CC_DEFAULTS)
            st.rerun()

    st.header("Compute vs Capabilities")
    # ══════════════════════════════════════════════════════════════════════
    # Section 1: Frontier compute growth, segmented
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Is compute slowing?")
    cap_date = (datetime(2028, 12, 31) if include_future else _today) + timedelta(days=_DAYS_2MO)
    frontier = _cc_trainflop_frontier(dc_all, cap_date)
    if not frontier:
        st.warning("No data-center data available.")
        return
    fits = _cc_segment_fits(frontier, _today)

    end_x = frontier[-1][0] + timedelta(days=30)
    x_start = datetime(2021, 1, 1)
    fig1 = go.Figure()
    if include_future:
        _dc_add_projection_band(fig1, _today, end_x)
    (a_x, a_y), (p_x, p_y) = _dc_split_at(frontier, _today, end_x)
    fig1.add_trace(go.Scatter(x=a_x, y=a_y, mode='lines',
                              line=dict(color='#1F77B4', width=3, shape='hv'),
                              hoverinfo='skip', showlegend=False))
    if p_x is not None:
        fig1.add_trace(go.Scatter(x=p_x, y=p_y, mode='lines',
                                  line=dict(color='#1F77B4', width=3, shape='hv', dash='dash'),
                                  hoverinfo='skip', showlegend=False))
    # Overlay each segment's log-linear fit to make the slope changes visible.
    seg_colors = ['#9467BD', '#2CA02C', '#D62728', '#FF7F0E']
    for i, fseg in enumerate(fits):
        c = seg_colors[i % len(seg_colors)]
        fig1.add_trace(go.Scatter(
            x=fseg['fit_x'], y=fseg['fit_y'], mode='lines',
            line=dict(color=c, width=2, dash='dot'),
            name=f"{fseg['label']}: ×{fseg['mult']:.1f}/yr",
            hovertext=[f"{fseg['label']}<br>×{fseg['mult']:.1f}/yr "
                       f"({fseg['doubling_mo']:.0f}-mo doubling)"] * 2,
            hoverinfo='text', showlegend=True))
    fig1.update_layout(**_dc_layout(
        True, "2mo train FLOP", x_start, end_x,
        y_range=_dc_yrange([v for _, v in frontier], True),
        show_legend=True))
    st.plotly_chart(fig1, use_container_width=True)

    # Segment growth table.
    rows_md = ["| Period | Growth | Doubling time | Milestones |",
               "|---|---|---|---|"]
    for fseg in fits:
        dbl = "—" if fseg['doubling_mo'] == float('inf') else f"{fseg['doubling_mo']:.1f} mo"
        rows_md.append(f"| {fseg['label']} | ×{fseg['mult']:.1f}/yr "
                       f"({fseg['slope_oom']:.2f} OOM/yr) | {dbl} | {fseg['n']} |")
    st.markdown("\n".join(rows_md))

    # ══════════════════════════════════════════════════════════════════════
    # Section 2: The exchange rate — how much capability per FLOP, and how fast
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
    st.caption(
        f"Based on {dec['n']} models reporting training compute. Row 2 is "
        "algorithmic efficiency (better architectures, data, RL, post-training, "
        "scaffolding) — i.e. “effective compute per real FLOP” going up.")

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
        yaxis=dict(title_text="Training compute (FLOP)", type='log',
                   gridcolor='rgba(0,0,0,0.12)', tickfont=dict(color='#222'),
                   title_font=dict(color='#222')))
    st.plotly_chart(figx, use_container_width=True)
    st.caption(
        "Each colored set is a fixed-capability band (ECI ± "
        f"{_CC_BAND_HALFWIDTH:.0f}); its matching dotted line is the within-band "
        "fit. The compute needed to stay in a band slopes *down* over time — that "
        "downward slope is the efficiency rate. Grey dots sit outside these bands.")

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
    st.caption(
        "Inverse regression `log10(FLOP) = α·ECI + βₜ·t + c` gives "
        f"−βₜ = {eff['g_inv']:.2f} OOM/yr (R² {eff['r2']:.2f}); the iso-ECI bands "
        f"give a median {eff['band_median']:.2f} OOM/yr. NB: ECI rewards "
        "reasoning/RL/post-training, so this is *total* capability efficiency, "
        "faster than pure-pretraining algorithmic efficiency.")

    # Mirror image: hold compute fixed, watch ECI climb.
    isoc = _cc_iso_compute(cc_rows)
    if isoc is not None:
        st.markdown(
            "**The mirror image — same compute, rising capability.** Flip the axes: "
            "hold the *compute budget* fixed and watch ECI climb. A model trained on "
            f"the same FLOP a year later scores about **+{isoc['eci_per_yr']:.0f} ECI "
            f"points** higher (range {isoc['lo']:.0f}–{isoc['hi']:.0f} across budgets).")
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
                name=f"~10^{c:.1f} FLOP  →  +{bseg['slope']:.0f} ECI/yr",
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
        st.caption(
            "Each colored set is a fixed *compute* band (log₁₀ FLOP ± "
            f"{_CC_CBAND_HALFWIDTH:.1f} dex); its dotted line slopes *up* — that's "
            "ECI gained per year at a constant compute budget. Same engine as the "
            "chart above, axes flipped.")

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
        st.caption(
            f"At a constant budget, a year's algorithmic progress adds ~"
            f"{isoc['eci_per_yr']:.0f} ECI — the same capability you'd otherwise "
            f"have to buy with ~{10 ** (isoc['eci_per_yr'] / epo):.0f}× more compute "
            f"(at {epo:.0f} ECI per ×10 FLOP). The mirror image of the table above: "
            "there, capability gets cheaper; here, the same spend buys more.")

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
    st.markdown(
        f"So **physical compute drives roughly a third to a half** of the "
        f"~{obs_slope:.0f} ECI-points/yr.")

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
    st.caption(
        f"The **hatched band** is the contested boundary (physical compute "
        f"**{share_lo*100:.0f}–{share_hi*100:.0f}%** of growth: low end = iso-compute "
        f"view, high end = iso-ECI view); the **solid line** is the central view "
        f"(~{share_mid*100:.0f}%). Algorithms are the rest. Frontier-model compute "
        f"growth (×{10**g_frontier:.1f}/yr) would push physical toward the high end.")

    st.warning(
        "**Caveats.** (1) The two engines aren't independent: algorithmic progress "
        "is *compute-fed* — you need GPUs to run the experiments — so a physical "
        "slowdown would likely drag the algorithmic rate too, making a stall "
        "*worse* than the ⅓–½ shown. (2) ECI bundles post-training/RL, so the "
        "efficiency rate is total-capability, not pretraining. (3) Labs rarely "
        "train small models just to re-hit old capability levels, so the cheap-"
        "model edge is sparse (Qwen, Kimi, distilled MoEs). (4) The 2mo-FLOP "
        "series is a capacity *ceiling*, not per-model training compute. Order-of-"
        "magnitude, not forecasts.")
    st.caption(
        "Data: Epoch AI Capabilities Index + Frontier Data Centers. The "
        "efficiency band spans the iso-ECI fit (compute on ECI+time) and the "
        "iso-compute fit (ECI on compute+time); these two OLS directions bracket "
        "the true rate, which errors-in-variables dilution puts between them. "
        "Frontier rates use the running-max-ECI subset.")

    # ══════════════════════════════════════════════════════════════════════
    # Section 3: ECI Forecasts — quarterly frontier projection to end of 2029
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("ECI Forecasts")
    _cc_eci_forecast(cc_rows, frontier, _today, obs_slope, g_recent, g_planned,
                     share_lo, share_mid, share_hi)

    # ══════════════════════════════════════════════════════════════════════
    # Section 4: US vs. China — the same decomposition read by country
    # ══════════════════════════════════════════════════════════════════════
    _cc_us_vs_china(cc_rows, _today)


# ── URL parameter persistence ────────────────────────────────────────────

_REV_DEFAULTS = {
    "rev_end_year": 2026,
    "rev_log_scale": True,
    "rev_2025_only": False,
    "rev_milestones": True,
    "rev_labels": True,
}
_REV_TRACKED_KEYS = list(_REV_DEFAULTS.keys()) + [
    "rev_proj_as_of", "oai_n_recent", "ant_n_recent",
    "oai_dt_lo", "oai_dt_hi", "ant_dt_lo", "ant_dt_hi",
]

_ECG_DEFAULTS = {"ecg_highlight": "None"}
_ECG_TRACKED_KEYS = list(_ECG_DEFAULTS.keys())

_PRINZ_DEFAULTS = {
    "prinz_group": "All models (one line)",
    "prinz_region": "Both",
    "prinz_labels": "Frontier only",
    "prinz_frontier": True,
}
_PRINZ_TRACKED_KEYS = list(_PRINZ_DEFAULTS.keys())

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
        (_PRINZ_TRACKED_KEYS, _PRINZ_DEFAULTS),
        (_EMP_RESET_KEYS, _EMP_DEFAULTS),
        (_REV_TRACKED_KEYS, _REV_DEFAULTS),
        (_ECG_TRACKED_KEYS, _ECG_DEFAULTS),
        (_DC_RESET_KEYS, _DC_DEFAULTS),
        (_CC_RESET_KEYS, _CC_DEFAULTS),
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


# ── Dispatch ─────────────────────────────────────────────────────────────

if not os.environ.get("_VP_TESTING"):
    _hydrate_session_from_url()

    if active_tab == "METR Horizon":
        render_metr()
    elif active_tab == "Epoch ECI":
        render_eci()
    elif active_tab == "Remote Labor Index":
        render_rli()
    elif active_tab == "Prinz":
        render_prinz()
    elif active_tab == "Revenue":
        render_revenue()
    elif active_tab == "Employment":
        render_employment()
    elif active_tab == "ECI Company Gap":
        render_eci_gap()
    elif active_tab == "Data Centers":
        render_data_centers()
    elif active_tab == "Compute vs Capabilities":
        render_compute_capabilities()

    _sync_session_to_url()

    # ── Share view: copy the (now state-synced) URL to the clipboard ──────
    import streamlit.components.v1 as _components
    with st.sidebar:
        st.markdown("---")
        _components.html(
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
            height=52,
        )
