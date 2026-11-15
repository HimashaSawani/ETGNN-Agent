import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment

def test_calipers(model_id, contrast, treat_cat, ctrl_cat):
    df = pd.read_csv(f'experiments/matched_c1_c2_audit/feasibility_{model_id}.csv')
    el = df[df['eligibility']]
    treat = el[el['prediction_category'] == treat_cat].reset_index(drop=True)
    ctrl = el[el['prediction_category'] == ctrl_cat].reset_index(drop=True)
    
    n_t, n_c = len(treat), len(ctrl)
    
    pooled_sd_m = float(np.sqrt((treat['margin'].var() + ctrl['margin'].var()) / 2))
    pooled_sd_e = float(np.sqrt((treat['candidate_edge_count'].var() + ctrl['candidate_edge_count'].var()) / 2))
    
    diff_m = np.abs(treat['margin'].values[:, None] - ctrl['margin'].values[None, :])
    diff_e = np.abs(treat['candidate_edge_count'].values[:, None] - ctrl['candidate_edge_count'].values[None, :])
    
    dist = np.sqrt((diff_m / pooled_sd_m)**2 + (diff_e / pooled_sd_e)**2)
    
    print(f"\n=== {model_id} {contrast} (Treat {treat_cat}={n_t}, Ctrl {ctrl_cat}={n_c}) ===")
    print(f"Pre-match Pooled SD: margin={pooled_sd_m:.4f}, edges={pooled_sd_e:.2f}")
    
    # Grid of candidate calipers
    calipers = [
        ("No Caliper", np.inf, np.inf),
        ("Caliper m<=0.10, e<=10", 0.10, 10),
        ("Caliper m<=0.075, e<=5", 0.075, 5),
        ("Caliper m<=0.05, e<=5", 0.05, 5),
        ("Caliper m<=0.25*SD, e<=5", 0.25 * pooled_sd_m, 5),
        ("Caliper m<=0.50*SD, e<=10", 0.50 * pooled_sd_m, 10),
    ]
    
    for label, cal_m, cal_e in calipers:
        admissible = (diff_m <= cal_m) & (diff_e <= cal_e)
        
        BIG = 10000.0
        # Augmented matrix to find maximum cardinality matching then min distance
        cost_mat = np.zeros((n_t + n_c, n_c + n_t))
        tl = np.full((n_t, n_c), BIG * 2)
        tl[admissible] = dist[admissible] - BIG
        cost_mat[:n_t, :n_c] = tl
        
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        
        matched_t = []
        matched_c = []
        for r, c in zip(row_ind, col_ind):
            if r < n_t and c < n_c and admissible[r, c]:
                matched_t.append(r)
                matched_c.append(c)
                
        n_m = len(matched_t)
        if n_m > 0:
            m_diff_m = diff_m[matched_t, matched_c]
            m_diff_e = diff_e[matched_t, matched_c]
            # SMD using PRE-MATCH pooled SD (standard practice: comparable denominator!)
            smd_m = float((treat.loc[matched_t, 'margin'].mean() - ctrl.loc[matched_c, 'margin'].mean()) / pooled_sd_m)
            smd_e = float((treat.loc[matched_t, 'candidate_edge_count'].mean() - ctrl.loc[matched_c, 'candidate_edge_count'].mean()) / pooled_sd_e)
            print(f"  {label:<26} | N_pairs={n_m:>3} ({n_m/min(n_t, n_c)*100:5.1f}%) | "
                  f"max diff_m={m_diff_m.max():.3f} (q95={np.quantile(m_diff_m, 0.95):.3f}) | "
                  f"max diff_e={m_diff_e.max():>3} (q95={np.quantile(m_diff_e, 0.95):.1f}) | "
                  f"SMD_m={smd_m:+.3f}, SMD_e={smd_e:+.3f}")
        else:
            print(f"  {label:<26} | N_pairs=0")

if __name__ == '__main__':
    test_calipers('etgnn_full', 'TP_FP', 'FP', 'TP')
    test_calipers('etgnn_full', 'TN_FN', 'FN', 'TN')
    test_calipers('matched_gat', 'TP_FP', 'FP', 'TP')
    test_calipers('matched_gat', 'TN_FN', 'FN', 'TN')
    test_calipers('etgnn_local', 'TP_FP', 'FP', 'TP')
    test_calipers('etgnn_local', 'TN_FN', 'FN', 'TN')
