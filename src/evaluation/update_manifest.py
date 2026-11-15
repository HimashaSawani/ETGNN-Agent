#!/usr/bin/env python3
import hashlib
from pathlib import Path

files = [
    'src/__init__.py',
    'src/models/__init__.py',
    'src/models/etgnn.py',
    'src/models/matched_gat.py',
    'src/models/time_encoding.py',
    'src/evaluation/__init__.py',
    'src/evaluation/run_main_c1_c2_benchmark.py',
    'src/evaluation/run_technical_pilot_benchmark.py',
    'configs/main_c1_c2_run_config.json',
    'data/processed/node_mapping.csv',
    'data/processed/validation_graph.pt',
    'results/checkpoints/etgnn.pt',
    'results/checkpoints/matched_gat.pt',
    'experiments/matched_c1_c2_audit/technical_pilot_sample.csv',
    'experiments/pilot_validation_audit/pilot_nodes.csv',
    'src/evaluation/test_c2_all_registers_synthetic.py',
]

lines = []
for f in files:
    p = Path(f)
    h = hashlib.sha256()
    with open(p, 'rb') as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    lines.append(f"{h.hexdigest()}  {p.stat().st_size:>10d}  {f}")

Path('manifest.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"Updated manifest.sha256 with {len(lines)} files.")
