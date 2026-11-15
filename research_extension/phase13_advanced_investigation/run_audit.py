"""
Audit script for Phase 13A
"""
import os
import sys
import json
import hashlib
import ast
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

def inspect_models():
    models_dir = ROOT / "src" / "models"
    model_files = list(models_dir.glob("*.py"))
    res = {}
    for mf in model_files:
        with open(mf, "r", encoding="utf-8") as f:
            code = f.read()
        try:
            tree = ast.parse(code)
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            res[mf.name] = {"classes": classes, "functions": funcs, "lines": len(code.splitlines())}
        except Exception as e:
            res[mf.name] = {"error": str(e)}
    return res

def inspect_checkpoints():
    ckpt_dir = ROOT / "results" / "checkpoints"
    ckpts = list(ckpt_dir.glob("*.*"))
    res = {}
    for c in ckpts:
        size = c.stat().st_size
        sha = hashlib.sha256(c.read_bytes()).hexdigest()
        res[c.name] = {"size": size, "sha256": sha}
    return res

def inspect_c1_c2():
    c1_path = ROOT / "results" / "tables" / "c1_response_summary.csv"
    c2_path = ROOT / "results" / "tables" / "c2_registers_bootstrap_summary.csv"
    res = {}
    if c1_path.exists():
        df1 = pd.read_csv(c1_path)
        res["c1"] = {"rows": len(df1), "columns": list(df1.columns), "data": df1.to_dict(orient="records")}
    if c2_path.exists():
        df2 = pd.read_csv(c2_path)
        res["c2"] = {"rows": len(df2), "columns": list(df2.columns), "data": df2.to_dict(orient="records")}
    return res

def inspect_phase4_phase10_phase12():
    res = {}
    p10_metrics = ROOT / "research_extension" / "phase10" / "PHASE10_MASTER_LOCAL_METRICS.csv"
    if p10_metrics.exists():
        df10 = pd.read_csv(p10_metrics)
        res["phase10_columns"] = list(df10.columns)
        res["phase10_models"] = df10.to_dict(orient="records")
    return res

def inspect_data_split():
    # check split files or logic
    data_dir = ROOT / "data"
    split_info = {}
    if data_dir.exists():
        files = list(data_dir.glob("**/*"))
        split_info["data_files"] = [str(f.relative_to(ROOT)) for f in files if f.is_file()]
    return split_info

if __name__ == "__main__":
    out = {
        "models": inspect_models(),
        "checkpoints": inspect_checkpoints(),
        "c1_c2": inspect_c1_c2(),
        "phase10_phase12": inspect_phase4_phase10_phase12(),
        "data_split": inspect_data_split()
    }
    out_file = ROOT / "research_extension" / "phase13_advanced_investigation" / "audit_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Audit completed and written to {out_file}")
