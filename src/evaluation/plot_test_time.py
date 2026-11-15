from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("results/tables/test_by_time_step.csv")
fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(data["time_step"], data["rf_auprc"],
        marker="o", linewidth=2, label="Random Forest")
ax.plot(data["time_step"], data["etgnn_auprc"],
        marker="s", linewidth=2, label="ETGNN")
ax.plot(data["time_step"], data["no_edge_auprc"],
        marker="^", linewidth=2, label="ETGNN without edges")
ax.plot(data["time_step"], data["illicit_prevalence"],
        linestyle="--", color="black", linewidth=1.5,
        label="Illicit prevalence (AP reference)")

ax.set(
    xlabel="Elliptic time step",
    ylabel="AUPRC / illicit prevalence",
    title="Held-out performance across time steps 41-49",
    xticks=range(41, 50),
    ylim=(0, 1.05),
)
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.text(
    0.01, 0.01,
    "Descriptive only; time steps have different illicit counts "
    "(step 46 has only 2).",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.045, 1, 1))

out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "test_auprc_by_time_step.png", dpi=220)
fig.savefig(out / "test_auprc_by_time_step.pdf")
print("Saved PNG and PDF to", out)

