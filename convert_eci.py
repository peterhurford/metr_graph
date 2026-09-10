"""Write Epoch's eci_scores.csv in the epoch_capabilities_index.csv layout the app reads.

eci_scores.csv has one row per model and no training-compute column, so
`Training compute (FLOP)` and `Confidence` are joined from Epoch's
all_ai_models.csv by model name.

    python3 convert_eci.py eci_scores.csv all_ai_models.csv epoch_capabilities_index.csv
"""
import csv
import sys

COLUMNS = ['Model version', 'ECI Score', 'Release date', 'Organization', 'Country',
           'Model accessibility', 'Training compute (FLOP)', 'Confidence',
           'Model name', 'Description', 'Display name']

# eci_scores.csv name -> all_ai_models.csv name, where the two spell one model
# differently. Each pairing reproduces the compute Epoch's former per-version
# export gave that model. Guarded by test_convert_eci.py.
ALIASES = {
    'DeepSeek V4 Pro 0813': 'DeepSeek-V4-Pro',
    'DeepSeek-V2 (MoE-236B, May 2024)': 'DeepSeek-V2 (MoE-236B)',
    'Grok-2 (Dec 2024)': 'Grok-2',
    'Kimi K2 (Jul 2025)': 'Kimi K2',
    'Qwen2.5-Coder-32B': 'Qwen2.5-Coder (32B)',
    'Qwen3-235B-A22B-Instruct (Jul 2025)': 'Qwen3-235B-A22B (Jul 2025)',
    'Qwen3-30B-A3B-Instruct (Jul 2025)': 'Qwen3-30B-A3B (Jul 2025)',
    'Qwen3-30B-A3B-Thinking (Jul 2025)': 'Qwen3-30B-A3B (Jul 2025)',
}


def convert(eci_rows, model_rows):
    models = {r['Model']: r for r in model_rows}
    out = []
    for r in eci_rows:
        name = r['Model']
        m = models.get(name) or models.get(ALIASES.get(name), {})
        out.append({
            'Model version': name,
            'ECI Score': r['eci'],
            'Release date': r['date'],
            'Organization': r['Organization'],
            'Country': r['Country (of organization)'],
            'Model accessibility': r['Model accessibility'],
            'Training compute (FLOP)': m.get('Training compute (FLOP)', ''),
            'Confidence': m.get('Confidence', ''),
            'Model name': name,
            'Description': '',
            'Display name': r['Display name'] or name,
        })
    return out


def main(eci_path, models_path, out_path):
    with open(eci_path, newline='') as f:
        eci = list(csv.DictReader(f))
    with open(models_path, newline='') as f:
        models = list(csv.DictReader(f))
    rows = convert(eci, models)
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
    names = {r['Model'] for r in eci}
    known = {r['Model'] for r in models}
    stale = [k for k, v in ALIASES.items() if k not in names or v not in known]
    print(f"{len(rows)} models, "
          f"{sum(bool(r['Training compute (FLOP)']) for r in rows)} with training compute")
    if stale:
        print("stale aliases:", stale)


if __name__ == '__main__':
    main(*sys.argv[1:4])
