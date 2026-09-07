"""Public declaration timeline, transcribed from the user's September 6, 2026 brief.

These are reported assessments, not independent capability measurements. The
framework labels are intentionally never converted to a common numeric scale.
"""
from pathlib import Path
import csv
import html
import io
import json
import textwrap

import plotly.graph_objects as go

AS_OF = '2026-09-06'
LABS = ['OpenAI', 'Anthropic', 'Google DeepMind']
COLORS = dict(zip(LABS, ['#10a37f', '#cc785c', '#4285f4']))
SYMBOLS = {
    'Assessment': 'circle',
    'Precaution': 'diamond-open',
    'Warning / alert': 'triangle-up-open',
    'Framework change': 'square',
    'Reassessment': 'x',
    'Operational update': 'star-open',
}
DEFAULTS = {'ft_lab': 'All labs', 'ft_domain': 'All domains',
            'ft_status': 'All event types', 'ft_search': '', 'ft_category': 'Cybersecurity'}


def load_events():
    return json.loads(Path(__file__).with_name('frontier_thresholds.json').read_text())['events']


def filter_events(events, lab='All labs', domain='All domains',
                  status='All event types', search=''):
    return [e for e in events
            if (lab == 'All labs' or e['lab'] == lab)
            and (domain == 'All domains' or domain in e['domains'])
            and (status == 'All event types' or e['status'] == status)
            and search.casefold() in ' '.join(str(v) for v in e.values()).casefold()]


def _wrap(value, width=85):
    return '<br>'.join(html.escape(s) for s in textwrap.wrap(value, width))


# Explicit domain-specific text prevents multi-domain source rows from leaking
# biological ratings into cyber panels (or vice versa).
CATEGORY_MILESTONES = {
    'Cybersecurity': {
        'ft-006': ('Medium', 'Deep Research', 'First Medium cybersecurity assessment under the original framework.'),
        'ft-012': ('High · precautionary', 'GPT-5.3-Codex', 'Treated as High without definitive evidence of fully automated end-to-end attacks.'),
        'ft-015': ('Critical · confirmed', 'Astra', 'Confirmed September 1. August 7 was the earlier precautionary trigger.'),
        'ft-039': ('Cyber Tier 1', 'Mythos 5.1', 'Assessed Cyber Tier 1 under the FCF, approaching Tier 2.'),
        'ft-043': ('Cyber alert', 'Gemini 2.5 Pro', 'March 25 checkpoint; documented by June 27. Cyber CCL not reached.'),
        'ft-050': ('Cyber alert continues', 'Gemini 3.1 Pro', 'Continued cyber alert, below the cyber CCL.'),
    },
    'Biological / chemical': {
        'ft-003': ('Medium CBRN', 'o1-preview / o1-mini', 'Greater assistance to biological experts; assessed Medium under the original framework.'),
        'ft-009': ('High · precautionary', 'ChatGPT agent', 'Treated as High despite lacking definitive evidence of novice uplift.'),
        'ft-022': ('ASL-3 · precautionary', 'Opus 4', 'Biological uncertainty triggered ASL-3 protections, not a definitive capability finding.'),
        'ft-038': ('Below next biological tier', 'Fable / Mythos 5.1', 'Still below the next chemical/biological tier under the revised RSP.'),
        'ft-044': ('Bio CCL uncertain', 'Deep Think', 'Biological alert reached; the biological CCL could not be ruled out.'),
        'ft-047': ('Bio CCL ruled out', 'Deep Think reassessment', 'Further analysis concluded the earlier biological CCL had not been reached.'),
        'ft-052': ('Biological alert returns', 'Gemini 3.7 Flash', 'Biological alert reached again, while remaining below the biological CCL.'),
    },
    'Autonomy / AI R&D': {
        'ft-005': ('Medium autonomy', 'o3-mini', 'Assessment under the old autonomy rubric, not recursive self-improvement.'),
        'ft-016': ('Below High self-improvement', 'Astra', 'Published assessment remained below High AI Self-Improvement.'),
        'ft-029': ('Near R&D threshold', 'Opus 4.6', 'Assessed below old AI R&D-4 despite a more tenuous rule-out case. Not ASL-4.'),
        'ft-037': ('Below automated-R&D threshold', 'August risk report', 'Increasing research contribution without a formal research-acceleration crossing under the revised definition.'),
    },
    'Persuasion': {
        'ft-002': ('Medium persuasion', 'GPT-4o', 'Text persuasion narrowly entered Medium; voice remained Low.'),
        'ft-040': ('Tier-2-range benchmark', 'Helpful-only Mythos 5.1', 'Evaluation warning only: the full harmful-manipulation threshold was not conclusively surpassed.'),
    },
}
# Keep the existing sparse chart labels; additional milestones get hoverable
# points and readable rows below, without crowding the timeline with text.
CHART_LABEL_IDS = {category: set(rows) for category, rows in CATEGORY_MILESTONES.items()}
CATEGORY_MILESTONES['Cybersecurity'].update({
    'ft-011': ('Below High; warning', 'GPT-5.2-Codex', 'OpenAI anticipated stronger successors and prepared cyber safeguards and trusted access. This was not the later High designation.'),
    'ft-013': ('Critical cannot be ruled out', 'Astra', 'August 7 precautionary trigger, not an affirmative Critical assessment. Heightened containment, weight protection, and monitoring restricted development.'),
    'ft-014': ('Training restrictions reported', 'August implementation update', 'Reported a two-week RL pause; the largest planned frontier RL run remained on hold. The response addressed both a security incident and potential Critical cyber capability.'),
    'ft-032': ('Restricted cyber deployment', 'Mythos Preview', 'Restricted deployment for cybersecurity work; the access arrangement did not itself establish an FCF cyber tier.'),
    'ft-034': ('Two access configurations', 'Fable 5 / Mythos 5', 'Fable was broadly released with tighter safeguards; less-restricted Mythos access was reserved for selected users. No new cyber tier was declared.'),
    'ft-052': ('Cyber alert maintained', 'Gemini 3.7 Flash', 'The cyber alert continued without an affirmative cyber CCL crossing.'),
    'ft-055': ('Restricted Cyber configuration', 'Gemini 3.8 Flash Cyber', 'Restricted Fairwind access, with no affirmative CCL declaration found in the announcement. The ordinary Flash assessment must not automatically be assigned to this configuration.'),
})
CATEGORY_MILESTONES['Biological / chemical'].update({
    'ft-008': ('Approaching High biology', 'Biology preparedness announcement', 'Preparations for more capable biological models and stronger safeguards, not a declaration that a named model had crossed High.'),
    'ft-010': ('High safeguards extended', 'GPT-5 Thinking', 'Extended the precautionary High biological approach to the new reasoning model; no higher threshold.'),
    'ft-016': ('High Bio/Chem retained', 'Astra system card', 'The release card confirmed High Biological and Chemical status.'),
    'ft-020': ('ASL-2; successor warning', 'Claude 3.7 Sonnet', 'ASL-3 was not activated for this release; stronger safeguards might be needed for successors.'),
    'ft-025': ('ASL-3 extended', 'Sonnet 4.5', 'Precautionary ASL-3 extension. This remained a model-specific safeguard classification.'),
    'ft-036': ('CB-1 treatment by default', 'August risk report', 'Most new models were treated as crossing CB-1 by default, with biological protections maintained. The lower threshold was no longer exceptional for each release.'),
})
CATEGORY_MILESTONES['Autonomy / AI R&D'].update({
    'ft-017': ('Research intern announcement', 'OpenAI research automation', 'The September 6 announcement did not change the published below-High AI Self-Improvement assessment.'),
    'ft-032': ('Sabotage threat model applies', 'Mythos Preview', 'Autonomy Threat Model 1 applied, while the higher automated-R&D threshold was assessed not crossed. Applicability is not full research-organization automation.'),
    'ft-035': ('R&D definition revised', 'July RSP revision', 'The automated-R&D threshold changed. Later below-threshold assessments must retain this policy version when compared with older AI R&D-4 findings.'),
})
CATEGORY_MILESTONES['Persuasion'].update({
    'ft-003': ('Medium persuasion retained', 'o1-preview / o1-mini', 'The original-framework persuasion assessment remained Medium before and after mitigations.'),
    'ft-007': ('Persuasion category retired', 'Preparedness v2', 'Persuasion ceased to be a tracked category in the revised framework. This was a policy change, not a lower capability assessment.'),
})

CATEGORY_SUMMARIES = {
    'Cybersecurity': ['Critical confirmed for Astra', 'Cyber Tier 1; approaching Tier 2', 'Cyber alerts; no confirmed cyber CCL'],
    'Biological / chemical': ['High biological safeguards', 'CB-1 protections; below the next tier', 'Biological alert; below biological CCL'],
    'Autonomy / AI R&D': ['Below High AI Self-Improvement', 'Below the revised automated-R&D threshold', 'No selected autonomy milestone in this timeline'],
    'Persuasion': ['Historical Medium rating under the old framework', 'Benchmark warning; full Tier 2 not established', 'No selected manipulation milestone in this timeline'],
}


def milestone_events(events, category='Cybersecurity', lab=None):
    curated = CATEGORY_MILESTONES[category]
    return [dict(e, threshold=curated[e['id']][0], model=curated[e['id']][1],
                 details=curated[e['id']][2], domains=[category])
            for e in events if e['id'] in curated and (lab is None or e['lab'] == lab)]


def timeline_figure(events, category='Cybersecurity', lab='OpenAI'):
    """A single lab and domain per chart; no overview slider or legend."""
    points = milestone_events(events, category, lab)
    fig = go.Figure(go.Scatter(
        x=[e['date'] for e in points], y=[0] * len(points),
        mode='markers+text', showlegend=False,
        text=[f"<b>{_wrap(e['threshold'], 26)}</b><br>{_wrap(e['model'], 26)}"
              if e['id'] in CHART_LABEL_IDS[category] else '' for e in points],
        textposition=[
            'top center' if sum(p['id'] in CHART_LABEL_IDS[category] for p in points[:i]) % 2 == 0
            else 'bottom center' for i in range(len(points))],
        textfont=dict(size=12), cliponaxis=False,
        marker=dict(size=[11 if e['id'] in CHART_LABEL_IDS[category] else 7 for e in points],
                    color=COLORS[lab]),
        customdata=[[_wrap(e[k], 65) for k in ('date_label', 'details')] for e in points],
        hovertemplate='%{customdata[0]}<br>%{customdata[1]}<extra></extra>'))
    fig.add_hline(y=0, line_width=1, line_color=COLORS[lab], opacity=0.25)
    fig.update_layout(
        height=195, margin=dict(l=95, r=115, t=55, b=30),
        showlegend=False, hoverlabel=dict(align='left'), dragmode=False,
        xaxis=dict(type='date', range=['2024-06-01', '2026-12-15'],
                   dtick='M6', tickformat='%b %Y', rangeslider=dict(visible=False),
                   showgrid=False, zeroline=False, fixedrange=True),
        yaxis=dict(visible=False, range=[-1, 1], fixedrange=True),
    )
    return fig


def events_csv(events):
    out = io.StringIO()
    fields = ['id', 'date', 'date_end', 'date_label', 'lab', 'model', 'domains',
              'framework', 'status', 'threshold', 'details', 'safeguards', 'provenance']
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(dict(e, domains='; '.join(e['domains'])) for e in events)
    return out.getvalue()


def _event_detail(st, event):
    st.markdown(f"**{event['threshold']}**")
    st.caption(f"{event['date_label']} · {event['framework']} · {event['status']}")
    st.write(event['details'])
    st.write('Response: ' + event['safeguards'])


def render(st):
    events = load_events()
    st.header('Frontier thresholds')
    st.caption('Public assessments through September 6, 2026. Each lab uses its own definitions; these are not equivalent risk levels.')
    if st.session_state.get('ft_category', 'Cybersecurity') not in CATEGORY_MILESTONES:
        st.session_state['ft_category'] = 'Cybersecurity'
    category = st.radio('Risk category', list(CATEGORY_MILESTONES), horizontal=True, key='ft_category')
    for lab, summary in zip(LABS, CATEGORY_SUMMARIES[category]):
        with st.container(border=True):
            st.markdown(f"### {lab}")
            st.caption(summary)
            points = milestone_events(events, category, lab)
            if points:
                st.plotly_chart(timeline_figure(events, category, lab), width='stretch',
                                config={'displayModeBar': False}, key=f'ft_chart_{lab}')
                st.dataframe([
                    {'Date': e['date_label'], 'Model / event': e['model'], 'Assessment / change': e['threshold']}
                    for e in points], hide_index=True, width='stretch')
                selected = st.selectbox(
                    'Explain a milestone', [e['id'] for e in points], index=len(points) - 1,
                    format_func=lambda ident, panel_points=points: next(
                        f"{e['date']} · {e['model']}" for e in panel_points if e['id'] == ident),
                    key=f'ft_detail_{category}_{lab}')
                detail = next(e for e in points if e['id'] == selected)
                st.caption(f"{detail['status']} · {detail['framework']}")
                st.write(detail['details'])
            else:
                st.caption('The supplied chronology contains no milestone selected for this category and lab.')
    with st.expander('Explore the full chronology · 55 events'):
        st.caption('Includes earlier Medium ratings, framework revisions, unchanged releases, and operational updates.')
        domains = sorted({d for e in events for d in e['domains']})
        def reset():
            st.session_state.update(DEFAULTS)
        options_by_key = [('ft_lab', ['All labs'] + LABS),
                          ('ft_domain', ['All domains'] + domains),
                          ('ft_status', ['All event types'] + list(SYMBOLS))]
        for key, options in options_by_key:
            if st.session_state.get(key, options[0]) not in options:
                st.session_state[key] = options[0]
        columns = st.columns(3)
        with columns[0]:
            lab = st.selectbox('Lab', options_by_key[0][1], key='ft_lab')
        with columns[1]:
            domain = st.selectbox('Domain', options_by_key[1][1], key='ft_domain')
        with columns[2]:
            status = st.selectbox('Event type', options_by_key[2][1], key='ft_status')
        search = st.text_input('Search timeline', key='ft_search')
        st.button('Reset timeline filters', on_click=reset)
        filtered = filter_events(events, lab, domain, status, search)
        if filtered:
            st.dataframe([{
                'Date': e['date_label'], 'Lab': e['lab'], 'Model / event': e['model'],
                'Declaration': e['threshold'], 'Event type': e['status'],
            } for e in filtered], hide_index=True, width='stretch')
            selected = st.selectbox('Read an event', [e['id'] for e in filtered],
                                   format_func=lambda ident: next(
                                       f"{e['date']} · {e['lab']} · {e['model']}"
                                       for e in filtered if e['id'] == ident))
            _event_detail(st, next(e for e in filtered if e['id'] == selected))
        else:
            st.info('No events match these filters. Reset the filters or broaden your search.')
        st.download_button('Download filtered timeline (CSV)', events_csv(filtered),
                           file_name='frontier_thresholds.csv', mime='text/csv')
    with st.expander('Definitions and source notes'):
        st.markdown(INTERPRETATION)
    st.caption('Source: supplied timeline, not independently verified.')


INTERPRETATION = '''
**OpenAI.** The 2023 scale used Low, Medium, High, and Critical across cyber,
CBRN, persuasion, and autonomy. The April 2025 revision removed Low/Medium
and tracks Bio/Chem, Cybersecurity, and AI Self-Improvement. Old persuasion
and autonomy scores cannot be carried into the new categories.

High biology concerns substantial assistance to a technically basic actor
creating a known biological or chemical threat; the first deployment designation
was precautionary. Critical cyber includes autonomous zero-day discovery and
exploitation across hardened systems, or novel end-to-end attacks on hardened
targets from a high-level objective. It is stronger than vulnerability finding.
Critical requires stopping further development until appropriate standards are
specified and then satisfying their controls; it is not a permanent training or
release ban. Whether those protections suffice is a separate question.

**Anthropic.** ASL-3 primarily denotes protections, also used as shorthand for
models requiring them. The supplied timeline contains no affirmative public
ASL-4 deployment declaration. Opus 4.6’s autonomy gray zone was assessed below
the old AI R&D-4 threshold. AI R&D-4 was not ASL-4: entry-level researcher
automation required ASL-3 security plus an affirmative misalignment-risk case;
AI R&D-5’s dramatic acceleration required at least ASL-4 security.

RSP v3 moved toward threat-specific risk reports and a safeguards roadmap.
CB-1 approximately concerns helping less-specialized actors with existing threats;
CB-2 concerns replacing scarce specialist expertise on more advanced threats.
Definitions changed in May and July 2026. The separate FCF’s cyber and manipulation
tiers are not ASLs or OpenAI classifications. A Tier-2-range manipulation benchmark
result did not establish the full harmful-manipulation threshold.

**Google DeepMind.** Early-warning alerts precede CCLs; TCLs track significant
risks at lower thresholds than the severe risks targeted by CCLs and are not
synonyms for alerts. The supplied material contains no affirmative public CCL
crossing. Deep Think’s August 2025 biological uncertainty was subsequently resolved
to “not reached.” Framework changes mean this does not demonstrate declining
capability. Biological alerts and chemical/radiological/nuclear TCL assessments
can concern different threat scenarios. Ordinary Flash assessments cannot be
assigned to a separately restricted Cyber configuration.

**Operational consequences.** Stronger security, monitoring, and differentiated
access have been common responses. OpenAI also reported actual training pauses;
its response addressed both a security incident and potential Critical capability.
Restricted access alone does not establish a threshold crossing. Research-automation
announcements are not formal self-improvement designations: the September 6
“automated research intern” announcement did not change Astra’s published
below-High AI Self-Improvement assessment.

**Provenance and dates.** This is a transcription of the user-provided public
frontier-threshold timeline through September 6, 2026, not an independent source
review. Dates are announcement/report dates unless explicitly noted. Month-only
entries occupy their full month. The Gemini 2.5 Pro checkpoint was March 25,
2025, documented by June 27; June 27 is the verified-card date in the supplied
brief, not necessarily its first disclosure. Reports retain their coverage dates.
Repeated unchanged releases and policy revisions are included to preserve context.
'''
