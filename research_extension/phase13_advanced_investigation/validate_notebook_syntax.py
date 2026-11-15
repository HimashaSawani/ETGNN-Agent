import json
import ast
from pathlib import Path

nb_path = Path("ETGNN_Phase13_Execution_Colab.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    first_line = cell['source'][0].strip() if cell['source'] else ""
    print(f"Cell {idx+1:2d} [{ctype:8s}]: {first_line[:60]}")
    if ctype == "code":
        code_lines = [l for l in cell['source'] if not l.strip().startswith("!") and not l.strip().startswith("%")]
        code_str = "".join(code_lines)
        try:
            ast.parse(code_str)
            print(f"         -> AST Syntax Valid ({len(code_lines)} lines)")
        except SyntaxError as e:
            print(f"         -> SYNTAX ERROR at line {e.lineno}: {e.msg}")
            raise e

print("\n[PASS] All notebook cells parsed and validated without any syntax errors!")

