import pandas as pd
import numpy as np

nodes = pd.read_csv('data/processed/nodes_with_masks.csv')
edges = pd.read_csv('data/raw/elliptic/elliptic_txs_edgelist.csv')

out_counts = edges['txId1'].value_counts()
in_counts = edges['txId2'].value_counts()

nodes['in_degree'] = nodes['txId'].map(in_counts).fillna(0).astype(int)
nodes['out_degree'] = nodes['txId'].map(out_counts).fillna(0).astype(int)
nodes['is_isolated'] = (nodes['in_degree'] == 0) & (nodes['out_degree'] == 0)

s43 = nodes[nodes['time_step'] == 43]
print("=== SNAPSHOT 43 AUDIT ===")
print("Total nodes in S43:", len(s43))
print("Labelled in S43:", int((s43['label'] != -1).sum()))
print("  - Illicit in S43:", int((s43['label'] == 1).sum()))
print("  - Licit in S43:", int((s43['label'] == 0).sum()))
print("Unlabelled in S43:", int((s43['label'] == -1).sum()))
print("ALL NODES S43:")
print("  - Mean in-degree:", float(s43['in_degree'].mean()))
print("  - Mean out-degree:", float(s43['out_degree'].mean()))
print("  - Zero in-degree count & pct:", int((s43['in_degree'] == 0).sum()), f"{float((s43['in_degree'] == 0).mean()*100):.2f}%")
print("  - Zero out-degree count & pct:", int((s43['out_degree'] == 0).sum()), f"{float((s43['out_degree'] == 0).mean()*100):.2f}%")
print("  - Completely isolated (in=0 AND out=0):", int(s43['is_isolated'].sum()), f"{float(s43['is_isolated'].mean()*100):.2f}%")

s43_illicit = s43[s43['label'] == 1]
print("ILLICIT ONLY S43 (N =", len(s43_illicit), "):")
print("  - Mean in-degree:", float(s43_illicit['in_degree'].mean()))
print("  - Mean out-degree:", float(s43_illicit['out_degree'].mean()))
print("  - Zero in-degree pct:", f"{float((s43_illicit['in_degree'] == 0).mean()*100):.2f}%")
print("  - Completely isolated pct:", f"{float(s43_illicit['is_isolated'].mean()*100):.2f}%")

s_base = nodes[nodes['time_step'] <= 32]
print("BASELINE (Snapshots 1-32, N =", len(s_base), "):")
print("  - Mean in-degree:", float(s_base['in_degree'].mean()))
print("  - Zero in-degree pct:", f"{float((s_base['in_degree'] == 0).mean()*100):.2f}%")
print("  - Completely isolated pct:", f"{float(s_base['is_isolated'].mean()*100):.2f}%")

