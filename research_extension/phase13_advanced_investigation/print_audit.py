import json
from pathlib import Path

p = Path("research_extension/phase13_advanced_investigation/audit_summary.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== MODELS ===")
for k, v in data["models"].items():
    print(f"  {k}: classes={v.get('classes')}, funcs={v.get('functions')}, lines={v.get('lines')}")

print("\n=== CHECKPOINTS ===")
for k, v in data["checkpoints"].items():
    print(f"  {k}: size={v['size']}, sha256={v['sha256'][:16]}...")

print("\n=== C1 SUMMARY ===")
if "c1" in data["c1_c2"]:
    print(f"  Rows: {data['c1_c2']['c1']['rows']}")
    for r in data["c1_c2"]["c1"]["data"]:
        print(f"    Cohort={r['Cohort']}: N={r['N']}, delta_adj_mean={r['delta_adj_mean']:.6f}, delta_raw_mean={r['delta_raw_mean']:.6f}")

print("\n=== C2 SUMMARY ===")
if "c2" in data["c1_c2"]:
    print(f"  Rows: {data['c1_c2']['c2']['rows']}")
    for r in data["c1_c2"]["c2"]["data"]:
        print(f"    Register={r['register']}: pairs={r['pairs_count']}, mean_diff={r['mean_diff']:.6f}, CI=[{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}], model={r['model']}")

print("\n=== PHASE 10 MODELS ===")
if "phase10_models" in data["phase10_phase12"]:
    print(f"  Total models in phase 10: {len(data['phase10_phase12']['phase10_models'])}")
    for m in data["phase10_phase12"]["phase10_models"][:10]:
        print(f"    {m.get('model_id')}: {m.get('model_name')} | AP={m.get('test_ap')} | F1={m.get('test_f1')}")

