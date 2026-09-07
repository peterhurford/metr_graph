"""Semantic regressions and real Streamlit checks for the supplied timeline."""
import csv
from datetime import date
import io

import frontier_thresholds as ft
from streamlit.testing.v1 import AppTest


def test_timeline_dates_and_distinctions():
    events = ft.load_events()
    assert len(events) == len({e['id'] for e in events}) == 55
    for e in events:
        assert date.fromisoformat(e['date']) <= date.fromisoformat(e['date_end']) <= date.fromisoformat(ft.AS_OF)
        assert e['status'] in ft.SYMBOLS
        assert all(e[k] for k in ['model', 'domains', 'framework', 'threshold', 'details', 'safeguards', 'provenance'])
    astra = ft.filter_events(events, lab='OpenAI', search='Astra')
    assert next(e for e in astra if e['date'] == '2026-08-07')['status'] == 'Precaution'
    assert next(e for e in astra if e['date'] == '2026-09-01')['status'] == 'Assessment'
    reversal = ft.filter_events(events, lab='Google DeepMind', status='Reassessment')
    assert len(reversal) == 1 and 'not reached' in reversal[0]['threshold']
    manipulation = ft.filter_events(events, lab='Anthropic', domain='Persuasion')
    assert len(manipulation) == 1 and manipulation[0]['status'] == 'Warning / alert'
    cyber = ft.filter_events(events, search='Gemini 3.8 Flash Cyber')
    assert cyber[0]['status'] == 'Operational update'


def test_chart_and_export_preserve_filter_and_date_qualifications():
    events = ft.filter_events(ft.load_events(), lab='Google DeepMind', domain='Cybersecurity')
    exported = list(csv.DictReader(io.StringIO(ft.events_csv(events))))
    assert len(exported) == len(events)
    checkpoint = next(e for e in exported if e['model'] == 'Gemini 2.5 Pro')
    assert checkpoint['date'] == '2025-06-27'
    assert 'March 25' in checkpoint['date_label']
    assert list(csv.DictReader(io.StringIO(ft.events_csv([])))) == []


def test_tab_filters_url_reset_and_empty_state():
    at = AppTest.from_file('visualize_projection.py', default_timeout=30)
    at.query_params.update(tab='thresholds', today_fwd='true', ft_lab='OpenAI', ft_domain='Cybersecurity', ft_search='Astra')
    at.run()
    assert not at.exception
    assert at.selectbox(key='ft_lab').value == 'OpenAI'
    assert len(at.get('plotly_chart')) == 3
    assert all(row == 'OpenAI' for row in at.dataframe[-1].value['Lab'])
    at.text_input(key='ft_search').set_value('no-such-event-xyz').run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 3  # Archive filters do not erase the overview.
    assert any('No events match' in v.value for v in at.info)
    next(b for b in at.button if b.label == 'Reset timeline filters').click().run()
    assert not at.exception
    assert at.selectbox(key='ft_lab').value == 'All labs'
    assert len(at.dataframe[-1].value) == 55
    assert 'ft_search' not in at.query_params
    at.selectbox(key='ft_status').set_value('Precaution').run()
    assert not at.exception
    assert at.query_params['ft_status'] == ['Precaution']
    assert set(at.dataframe[-1].value['Event type']) == {'Precaution'}
    at.radio(key='_active_tab').set_value('Revenue').run()
    assert not at.exception
    at.radio(key='_active_tab').set_value('Frontier Thresholds').run()
    assert not at.exception
    assert at.selectbox(key='ft_status').value == 'Precaution'


def test_invalid_url_filters_recover():
    at = AppTest.from_file('visualize_projection.py', default_timeout=30)
    at.query_params.update(tab='thresholds', ft_lab='unknown', ft_domain='unknown', ft_status='unknown')
    at.run()
    assert not at.exception
    assert len(at.dataframe[-1].value) == 55


def test_category_panels_never_mix_domains():
    events = ft.load_events()
    for category in ft.CATEGORY_MILESTONES:
        for lab in ft.LABS:
            points = ft.milestone_events(events, category, lab)
            assert all(e['domains'] == [category] and e['lab'] == lab for e in points)
            fig = ft.timeline_figure(events, category, lab)
            assert len(fig.data) == 1
            assert len(fig.data[0].x) == len(points)
            assert fig.layout.xaxis.rangeslider.visible is False
            assert fig.layout.showlegend is False
    cyber = ft.milestone_events(events, 'Cybersecurity')
    assert all('biolog' not in e['details'].lower() for e in cyber)
    biology = ft.milestone_events(events, 'Biological / chemical')
    assert next(e for e in biology if e['id'] == 'ft-047')['status'] == 'Reassessment'


def test_category_switches_and_collapsed_details():
    at = AppTest.from_file('visualize_projection.py', default_timeout=30)
    at.query_params['tab'] = 'thresholds'
    at.run()
    assert not at.exception
    assert len(at.get('plotly_chart')) == 3
    assert all(not expander.proto.expanded for expander in at.expander)
    for category, count in [('Biological / chemical', 3), ('Autonomy / AI R&D', 2), ('Persuasion', 2)]:
        at.radio(key='ft_category').set_value(category).run()
        assert not at.exception
        assert len(at.get('plotly_chart')) == count
        assert at.query_params['ft_category'] == [category]


def test_added_context_is_visible_without_crowding_chart_labels():
    events = ft.load_events()
    points = ft.milestone_events(events, 'Cybersecurity', 'OpenAI')
    assert {'ft-013', 'ft-014', 'ft-015'} <= {e['id'] for e in points}
    fig = ft.timeline_figure(events)
    assert len(fig.data[0].x) == 6
    assert sum(bool(label) for label in fig.data[0].text) == 3
    at = AppTest.from_file('visualize_projection.py', default_timeout=30)
    at.query_params['tab'] = 'thresholds'
    at.run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 6
    key = 'ft_detail_Cybersecurity_OpenAI'
    at.selectbox(key=key).set_value('ft-013').run()
    assert not at.exception
    assert any('August 7 precautionary trigger' in text.value for text in at.markdown)
