import csv
import os

import convert_eci as ce


def _eci(name):
    return {'Model': name, 'Display name': name, 'eci': '150.0', 'date': '2026-01-01',
            'Organization': 'OpenAI', 'Country (of organization)': 'United States of America',
            'Model accessibility': 'API access'}


def test_writes_the_layout_the_app_reads():
    [row] = ce.convert([_eci('GPT-X')], [{'Model': 'GPT-X', 'Training compute (FLOP)': '1e26',
                                          'Confidence': 'Confident'}])
    assert list(row) == ce.COLUMNS
    assert (row['ECI Score'], row['Release date'], row['Country'], row['Model name'],
            row['Training compute (FLOP)']) == (
        '150.0', '2026-01-01', 'United States of America', 'GPT-X', '1e26')


def test_alias_joins_compute_across_a_rename():
    [row] = ce.convert([_eci('Kimi K2 (Jul 2025)')],
                       [{'Model': 'Kimi K2', 'Training compute (FLOP)': '2.976e+24'}])
    assert row['Training compute (FLOP)'] == '2.976e+24'


def test_every_alias_resolved_in_the_committed_csv():
    # An alias whose model Epoch dropped or renamed again is dead weight; one
    # whose target stopped resolving silently drops a point from the compute fit.
    path = os.path.join(os.path.dirname(__file__), 'epoch_capabilities_index.csv')
    rows = {r['Model name']: r for r in csv.DictReader(open(path))}
    for name in ce.ALIASES:
        assert name in rows, name
        assert rows[name]['Training compute (FLOP)'], name
