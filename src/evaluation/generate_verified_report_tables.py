import os
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment


def calc_smd(treat_vals, ctrl_vals, pooled_sd_pre):
    if pooled_sd_pre == 0 or np.isnan(pooled_sd_pre):
        return 0.0
    return float((np.mean(treat_vals) - np.mean(ctrl_vals)) / pooled_sd_pre)


def calc_variance_ratio(treat_vals, ctrl_vals):
    v_t = np.var(treat_vals, ddof=1)
    v_c = np.var(ctrl_vals, ddof=1)
    if v_c == 0:
        return float("inf") if v_t > 0 else 1.0
    return float(v_t / v_c)


def main():
    print("=== AUTOMATED VERIFICATION AND REPORT TABLE GENERATOR ===")
    out_dir = Path("experiments/matched_c1_c2_audit")
    
    # 1. Row-by-row assertion of total_degree identity on feasibility CSVs
    print("\n--- 1. Feasibility CSV Row-by-Row Assertions ---")
    feas_files = {
        "ETGNN_full": out_dir / "feasibility_etgnn_full.csv",
        "Matched_GAT": out_dir / "feasibility_matched_gat.csv",
        "ETGNN_local": out_dir / "feasibility_etgnn_local.csv",
    }
    
    for m_name, fpath in feas_files.items():
        df = pd.read_csv(fpath)
        violations = int((df["total_degree"] != df["in_degree"] + df["out_degree"]).sum())
        assert violations == 0, f"Violation in {fpath.name}: {violations} rows have total_degree != in_degree + out_degree!"
        print(f"[{m_name}] {len(df)} rows verified: total_degree == in_degree + out_degree (0 violations).")

    # 2. Assertions on Matched Pair CSVs
    print("\n--- 2. Matched Pair Row-by-Row Assertions & Hopcroft-Karp Cardinality ---")
    caliper_m = 0.05
    caliper_e = 5
    
    verified_balance_records = []
    difference_distribution_records = []
    assertion_logs = []
    
    covariates_list = ["margin", "candidate_edge_count", "in_degree", "out_degree", "total_degree"]
    
    for model_id in ["ETGNN_full", "Matched_GAT", "ETGNN_local"]:
        df_feas = pd.read_csv(feas_files[model_id])
        eligible_df = df_feas[df_feas["eligibility"]].copy().reset_index(drop=True)
        
        for contrast_name, (treat_cat, ctrl_cat) in [("TP_FP", ("FP", "TP")), ("TN_FN", ("FN", "TN"))]:
            treat_nodes = eligible_df[eligible_df["prediction_category"] == treat_cat].copy().reset_index(drop=True)
            ctrl_nodes = eligible_df[eligible_df["prediction_category"] == ctrl_cat].copy().reset_index(drop=True)
            
            n_t = len(treat_nodes)
            n_c = len(ctrl_nodes)
            
            # Hopcroft-Karp Exact Maximum Cardinality Verification
            diff_m = np.abs(treat_nodes["margin"].values[:, None] - ctrl_nodes["margin"].values[None, :])
            diff_e = np.abs(treat_nodes["candidate_edge_count"].values[:, None] - ctrl_nodes["candidate_edge_count"].values[None, :])
            admissible = (diff_m <= caliper_m) & (diff_e <= caliper_e)
            
            adj = csr_matrix(admissible.astype(int))
            exact_max_card = int((maximum_bipartite_matching(adj, perm_type="column") >= 0).sum())
            
            # Load actual saved matched pairs
            pairs_p = out_dir / f"caliper_pairs_{model_id.lower()}_{contrast_name.lower()}.csv"
            pairs_df = pd.read_csv(pairs_p)
            actual_card = len(pairs_df)
            
            # Assertions
            assert actual_card == exact_max_card, f"Cardinality mismatch in {pairs_p.name}: {actual_card} vs exact max {exact_max_card}"
            assert pairs_df["treat_node_index"].nunique() == actual_card, f"Repeated treat nodes in {pairs_p.name}!"
            assert pairs_df["ctrl_node_index"].nunique() == actual_card, f"Repeated ctrl nodes in {pairs_p.name}!"
            assert len(set(pairs_df["treat_node_index"]).intersection(set(pairs_df["ctrl_node_index"]))) == 0, f"Treat/ctrl overlap in {pairs_p.name}!"
            assert (pairs_df["abs_diff_margin"] <= caliper_m).all(), f"Margin caliper violated in {pairs_p.name}!"
            assert (pairs_df["abs_diff_candidate_edges"] <= caliper_e).all(), f"Edge caliper violated in {pairs_p.name}!"
            assert (pairs_df["treat_total_degree"] == pairs_df["treat_in_degree"] + pairs_df["treat_out_degree"]).all(), "Treat total degree mismatch!"
            assert (pairs_df["ctrl_total_degree"] == pairs_df["ctrl_in_degree"] + pairs_df["ctrl_out_degree"]).all(), "Ctrl total degree mismatch!"
            
            print(f"[{model_id} {contrast_name}] Verified {actual_card} pairs: matches Hopcroft-Karp max cardinality ({exact_max_card}), zero repeated nodes, strict unrounded calipers satisfied.")
            assertion_logs.append({
                "model_id": model_id,
                "contrast": contrast_name,
                "treat_available": n_t,
                "ctrl_available": n_c,
                "matched_pairs": actual_card,
                "hopcroft_karp_max_card": exact_max_card,
                "unique_treat_nodes": pairs_df["treat_node_index"].nunique(),
                "unique_ctrl_nodes": pairs_df["ctrl_node_index"].nunique(),
                "caliper_margin_max_diff": round(float(pairs_df["abs_diff_margin"].max()), 4),
                "caliper_edges_max_diff": int(pairs_df["abs_diff_candidate_edges"].max()),
                "total_degree_identity_passed": True,
            })
            
            # Difference distributions
            diff_m_vals = pairs_df["abs_diff_margin"].values
            diff_e_vals = pairs_df["abs_diff_candidate_edges"].values
            difference_distribution_records.append({
                "model_id": model_id,
                "contrast": contrast_name,
                "n_pairs": actual_card,
                "diff_margin_mean": round(float(np.mean(diff_m_vals)), 4),
                "diff_margin_median": round(float(np.median(diff_m_vals)), 4),
                "diff_margin_q75": round(float(np.quantile(diff_m_vals, 0.75)), 4),
                "diff_margin_q95": round(float(np.quantile(diff_m_vals, 0.95)), 4),
                "diff_margin_max": round(float(np.max(diff_m_vals)), 4),
                "diff_edges_mean": round(float(np.mean(diff_e_vals)), 2),
                "diff_edges_median": round(float(np.median(diff_e_vals)), 2),
                "diff_edges_q75": round(float(np.quantile(diff_e_vals, 0.75)), 2),
                "diff_edges_q95": round(float(np.quantile(diff_e_vals, 0.95)), 2),
                "diff_edges_max": int(np.max(diff_e_vals)),
            })
            
            # Balance statistics across all 5 covariates
            pooled_sds_pre = {}
            for cov in covariates_list:
                s_t = np.var(treat_nodes[cov], ddof=1)
                s_c = np.var(ctrl_nodes[cov], ddof=1)
                pooled_sds_pre[cov] = float(np.sqrt((s_t + s_c) / 2.0))
                
            # Verify sum of means for degrees in pre-match
            pre_t_in = float(treat_nodes["in_degree"].mean())
            pre_t_out = float(treat_nodes["out_degree"].mean())
            pre_t_tot = float(treat_nodes["total_degree"].mean())
            assert abs((pre_t_in + pre_t_out) - pre_t_tot) < 1e-6, "Pre-treat degree sum violation!"
            
            pre_c_in = float(ctrl_nodes["in_degree"].mean())
            pre_c_out = float(ctrl_nodes["out_degree"].mean())
            pre_c_tot = float(ctrl_nodes["total_degree"].mean())
            assert abs((pre_c_in + pre_c_out) - pre_c_tot) < 1e-6, "Pre-ctrl degree sum violation!"
            
            # Post-match degree sum verification
            post_t_in = float(pairs_df["treat_in_degree"].mean())
            post_t_out = float(pairs_df["treat_out_degree"].mean())
            post_t_tot = float(pairs_df["treat_total_degree"].mean())
            assert abs((post_t_in + post_t_out) - post_t_tot) < 1e-6, "Post-treat degree sum violation!"
            
            post_c_in = float(pairs_df["ctrl_in_degree"].mean())
            post_c_out = float(pairs_df["ctrl_out_degree"].mean())
            post_c_tot = float(pairs_df["ctrl_total_degree"].mean())
            assert abs((post_c_in + post_c_out) - post_c_tot) < 1e-6, "Post-ctrl degree sum violation!"
            
            for cov in covariates_list:
                pre_t_m = float(treat_nodes[cov].mean())
                pre_c_m = float(ctrl_nodes[cov].mean())
                pre_smd = calc_smd(treat_nodes[cov], ctrl_nodes[cov], pooled_sds_pre[cov])
                pre_vr = calc_variance_ratio(treat_nodes[cov], ctrl_nodes[cov])
                
                post_t_col = "treat_candidate_edges" if cov == "candidate_edge_count" else f"treat_{cov}"
                post_c_col = "ctrl_candidate_edges" if cov == "candidate_edge_count" else f"ctrl_{cov}"
                
                post_t_vals = pairs_df[post_t_col]
                post_c_vals = pairs_df[post_c_col]
                post_t_m = float(post_t_vals.mean())
                post_c_m = float(post_c_vals.mean())
                post_smd = calc_smd(post_t_vals, post_c_vals, pooled_sds_pre[cov])
                post_vr = calc_variance_ratio(post_t_vals, post_c_vals)
                
                verified_balance_records.append({
                    "model_id": model_id,
                    "contrast": contrast_name,
                    "covariate": cov,
                    "n_treat_available": n_t,
                    "n_ctrl_available": n_c,
                    "n_matched_pairs": actual_card,
                    "treat_coverage_pct": round(actual_card / n_t * 100, 1),
                    "ctrl_coverage_pct": round(actual_card / n_c * 100, 1),
                    "pooled_sd_pre": round(pooled_sds_pre[cov], 4),
                    "pre_treat_mean": round(pre_t_m, 4),
                    "pre_ctrl_mean": round(pre_c_m, 4),
                    "pre_smd": round(pre_smd, 3),
                    "pre_vr": round(pre_vr, 3),
                    "post_treat_mean": round(post_t_m, 4),
                    "post_ctrl_mean": round(post_c_m, 4),
                    "post_smd": round(post_smd, 3),
                    "post_vr": round(post_vr, 3) if not np.isinf(post_vr) else np.inf,
                })

    df_bal = pd.DataFrame(verified_balance_records)
    df_bal.to_csv(out_dir / "verified_caliper_balance_diagnostics.csv", index=False)
    
    df_diffs = pd.DataFrame(difference_distribution_records)
    df_diffs.to_csv(out_dir / "verified_caliper_within_pair_differences.csv", index=False)
    
    df_assert = pd.DataFrame(assertion_logs)
    df_assert.to_csv(out_dir / "assertion_verification_results.csv", index=False)
    print("\nSaved verified_caliper_balance_diagnostics.csv and assertion_verification_results.csv.")

    # 3. Degree-Balanced Sensitivity Match Evaluation
    print("\n--- 3. Degree-Balanced Sensitivity Match ---")
    deg_sens_records = []
    for model_id in ["ETGNN_full", "Matched_GAT", "ETGNN_local"]:
        df_feas = pd.read_csv(feas_files[model_id])
        eligible_df = df_feas[df_feas["eligibility"]].copy().reset_index(drop=True)
        
        for contrast_name, (treat_cat, ctrl_cat) in [("TP_FP", ("FP", "TP")), ("TN_FN", ("FN", "TN"))]:
            treat_nodes = eligible_df[eligible_df["prediction_category"] == treat_cat].copy().reset_index(drop=True)
            ctrl_nodes = eligible_df[eligible_df["prediction_category"] == ctrl_cat].copy().reset_index(drop=True)
            n_t, n_c = len(treat_nodes), len(ctrl_nodes)
            
            diff_m = np.abs(treat_nodes["margin"].values[:, None] - ctrl_nodes["margin"].values[None, :])
            diff_e = np.abs(treat_nodes["candidate_edge_count"].values[:, None] - ctrl_nodes["candidate_edge_count"].values[None, :])
            diff_out = np.abs(treat_nodes["out_degree"].values[:, None] - ctrl_nodes["out_degree"].values[None, :])
            
            pooled_sd_m = float(np.sqrt((treat_nodes["margin"].var() + ctrl_nodes["margin"].var()) / 2.0))
            pooled_sd_e = float(np.sqrt((treat_nodes["candidate_edge_count"].var() + ctrl_nodes["candidate_edge_count"].var()) / 2.0))
            pooled_sd_out = float(np.sqrt((treat_nodes["out_degree"].var() + ctrl_nodes["out_degree"].var()) / 2.0))
            
            # Caliper: margin <= 0.05, edges <= 5, AND out_degree diff == 0 (exact out-degree match)
            admissible_sens = (diff_m <= 0.05) & (diff_e <= 5) & (diff_out == 0)
            
            # Hopcroft-Karp cardinality
            adj_sens = csr_matrix(admissible_sens.astype(int))
            exact_sens_card = int((maximum_bipartite_matching(adj_sens, perm_type="column") >= 0).sum())
            
            # Min cost assignment
            dist_sens = np.sqrt(
                (diff_m / pooled_sd_m)**2 + (diff_e / pooled_sd_e)**2 + (diff_out / max(pooled_sd_out, 1e-6))**2
            )
            BIG = 10000.0
            cost_mat = np.zeros((n_t + n_c, n_c + n_t))
            tl = np.full((n_t, n_c), BIG * 2.0)
            tl[admissible_sens] = dist_sens[admissible_sens] - BIG
            cost_mat[:n_t, :n_c] = tl
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            
            sens_pairs = []
            for r, c in zip(row_ind, col_ind):
                if r < n_t and c < n_c and admissible_sens[r, c]:
                    t_row = treat_nodes.iloc[r]
                    c_row = ctrl_nodes.iloc[c]
                    sens_pairs.append({
                        "pair_id": f"{model_id}_{contrast_name}_deg_sens_{len(sens_pairs)+1:03d}",
                        "treat_node_index": int(t_row["node_index"]),
                        "treat_txId": int(t_row["transaction_id"]),
                        "treat_margin": float(t_row["margin"]),
                        "treat_candidate_edges": int(t_row["candidate_edge_count"]),
                        "treat_in_degree": int(t_row["in_degree"]),
                        "treat_out_degree": int(t_row["out_degree"]),
                        "treat_total_degree": int(t_row["total_degree"]),
                        "ctrl_node_index": int(c_row["node_index"]),
                        "ctrl_txId": int(c_row["transaction_id"]),
                        "ctrl_margin": float(c_row["margin"]),
                        "ctrl_candidate_edges": int(c_row["candidate_edge_count"]),
                        "ctrl_in_degree": int(c_row["in_degree"]),
                        "ctrl_out_degree": int(c_row["out_degree"]),
                        "ctrl_total_degree": int(c_row["total_degree"]),
                    })
                    
            sens_df = pd.DataFrame(sens_pairs)
            sens_df.to_csv(out_dir / f"degree_sensitivity_pairs_{model_id.lower()}_{contrast_name.lower()}.csv", index=False)
            
            # Post balance for sensitivity match
            post_smd_m = calc_smd(sens_df["treat_margin"], sens_df["ctrl_margin"], pooled_sd_m)
            post_smd_e = calc_smd(sens_df["treat_candidate_edges"], sens_df["ctrl_candidate_edges"], pooled_sd_e)
            post_smd_out = calc_smd(sens_df["treat_out_degree"], sens_df["ctrl_out_degree"], pooled_sd_out)
            pooled_sd_tot = float(np.sqrt((treat_nodes["total_degree"].var() + ctrl_nodes["total_degree"].var()) / 2.0))
            post_smd_tot = calc_smd(sens_df["treat_total_degree"], sens_df["ctrl_total_degree"], pooled_sd_tot)
            
            deg_sens_records.append({
                "model_id": model_id,
                "contrast": contrast_name,
                "n_matched_pairs": len(sens_df),
                "treat_coverage_pct": round(len(sens_df) / n_t * 100, 1),
                "ctrl_coverage_pct": round(len(sens_df) / n_c * 100, 1),
                "post_smd_margin": round(post_smd_m, 3),
                "post_smd_edges": round(post_smd_e, 3),
                "post_smd_out_degree": round(post_smd_out, 3),
                "post_smd_total_degree": round(post_smd_tot, 3),
            })
            print(f"[{model_id} {contrast_name}] Degree-balanced sensitivity: {len(sens_df)} pairs. Post SMD out_degree = {post_smd_out:.3f}, total_degree = {post_smd_tot:.3f}, margin = {post_smd_m:.3f}, edges = {post_smd_e:.3f}.")

    df_sens = pd.DataFrame(deg_sens_records)
    df_sens.to_csv(out_dir / "degree_sensitivity_balance_summary.csv", index=False)
    
    # 4. Generate Automatic Markdown Report Tables
    print("\n--- 4. Generating Automated Markdown Report Tables ---")
    gen_md = []
    gen_md.append("### Verified Balance Table (Directly Generated from Data)\n")
    gen_md.append("| Model | Contrast | Covariate | Pre Pooled SD | Pre Treat Mean | Pre Ctrl Mean | Pre SMD | Post Treat Mean | Post Ctrl Mean | Post SMD | Pre VR | Post VR |")
    gen_md.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for _, r in df_bal.iterrows():
        vr_post_str = f"{r['post_vr']:.3f}" if not np.isinf(r["post_vr"]) else "$\infty$"
        gen_md.append(
            f"| **`{r['model_id']}`** | **{r['contrast']}** | **{r['covariate']}** | "
            f"{r['pooled_sd_pre']:.4f} | {r['pre_treat_mean']:.3f} | {r['pre_ctrl_mean']:.3f} | "
            f"**{r['pre_smd']:+.3f}** | {r['post_treat_mean']:.3f} | {r['post_ctrl_mean']:.3f} | "
            f"**{r['post_smd']:+.3f}** | {r['pre_vr']:.3f} | {vr_post_str} |"
        )
        
    gen_md.append("\n### Verified Degree-Balanced Sensitivity Table\n")
    gen_md.append("| Model | Contrast | Matched Pairs | Treat Coverage | Ctrl Coverage | Post SMD Margin | Post SMD Edges | Post SMD Out-Degree | Post SMD Total-Degree |")
    gen_md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_sens.iterrows():
        gen_md.append(
            f"| **`{r['model_id']}`** | **{r['contrast']}** | **{r['n_matched_pairs']}** | "
            f"{r['treat_coverage_pct']:.1f}% | {r['ctrl_coverage_pct']:.1f}% | "
            f"**{r['post_smd_margin']:+.3f}** | **{r['post_smd_edges']:+.3f}** | "
            f"**{r['post_smd_out_degree']:+.3f}** | **{r['post_smd_total_degree']:+.3f}** |"
        )
        
    table_snippet_path = out_dir / "automated_report_tables.md"
    table_snippet_path.write_text("\n".join(gen_md), encoding="utf-8")
    print(f"Generated automated report markdown snippet: {table_snippet_path}")
    print("\nPipeline finished successfully.")

if __name__ == "__main__":
    main()
