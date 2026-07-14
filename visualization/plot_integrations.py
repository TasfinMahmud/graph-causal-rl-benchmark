import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = r"../results/integration_results"
os.makedirs(RESULTS_PATH, exist_ok=True)

# 1. Plot Integration A (GNN + Causal)
df_a = pd.read_csv(os.path.join(RESULTS_PATH, "integration_a_metrics.csv"))

plt.figure(figsize=(10, 6))
sns.barplot(data=df_a, x='Dataset', y='Score', hue='Model')
# plt.title('Integration A: CATE Variance Reduction (GNN vs Raw Features)', fontsize=14, pad=15)
plt.ylabel('CATE Variance (Lower is better)', fontsize=12)
plt.yscale('log')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, 'integration_a_plot.png'), dpi=300)
paper_dir = r"../results"
plt.savefig(os.path.join(paper_dir, 'integration_a.png'), dpi=300)
plt.close()

# 2. Plot Integration B (Causal + Offline RL)
df_b = pd.read_csv(os.path.join(RESULTS_PATH, "integration_b_metrics.csv"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# OBD Subplot
df_obd = df_b[df_b['Dataset'] == 'OBD']
sns.barplot(data=df_obd, x='Dataset', y='Score', hue='Model', palette=['#ff7f0e', '#1f77b4'], ax=ax1)
ax1.set_title('OBD (SNIPW CTR)')
ax1.set_ylabel('SNIPW CTR')
ax1.get_legend().remove()

# KUAIREC Subplot
df_kuai = df_b[df_b['Dataset'] == 'KUAIREC']
sns.barplot(data=df_kuai, x='Dataset', y='Score', hue='Model', palette=['#ff7f0e', '#1f77b4'], ax=ax2)
ax2.set_title('KuaiRec (Hit Rate)')
ax2.set_ylabel('Hit Rate')
ax2.set_ylim(0, 1.0)
ax2.legend(title='Offline RL Agent', loc='upper right')

# plt.suptitle('Integration B: Causal BCQ vs Naive DQN', fontsize=14)
plt.tight_layout()

# Save to integration_results folder
plt.savefig(os.path.join(RESULTS_PATH, 'integration_b_plot.png'), dpi=300)
# Save directly to the IEEE template folder
paper_dir = r"../results"
plt.savefig(os.path.join(paper_dir, 'integration_b.png'), dpi=300)
plt.close()

print("Plots generated successfully.")
