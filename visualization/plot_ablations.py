import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = r"../data"
OUTPUT_DIR = r"../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_ablation_study(dataset_name):
    csv_file = os.path.join(DATA_DIR, f"tournament_{dataset_name}_metrics.csv")
    if not os.path.exists(csv_file):
        logging.error(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # We will filter for SNIPW_CTR or GroundTruth_HitRate
    metric_filter = "SNIPW_CTR" if dataset_name == "obd" else "GroundTruth_HitRate"
    df_metric = df[df['Metric'] == metric_filter]

    # Extract the scores we need
    # We map Causal BCQ -> "BCQ + LightGCN (Core)"
    # We map DiscreteBCQ -> "BCQ (Raw Features)"
    # We map DQN -> "Unconstrained DQN"
    
    score_dict = {}
    for _, row in df_metric.iterrows():
        score_dict[row['Model']] = row['Score']
        
    if "Causal BCQ" not in score_dict or "DiscreteBCQ" not in score_dict or "DQN" not in score_dict:
        logging.warning(f"Missing models for ablation in {dataset_name}.")
        return

    # Prepare data for Ablation 1 (GNN vs No GNN)
    ablation1_data = {
        "Configuration": ["BCQ + LightGCN", "BCQ (Raw Features Only)"],
        "Score": [score_dict.get("Causal BCQ", 0), score_dict.get("DiscreteBCQ", 0)]
    }
    
    # Prepare data for Ablation 2 (BCQ vs DQN)
    ablation2_data = {
        "Configuration": ["BCQ (Constrained)", "DQN (Unconstrained)"],
        "Score": [score_dict.get("Causal BCQ", 0), score_dict.get("DQN", 0)]
    }

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    
    # Plot Ablation 1
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Configuration", y="Score", data=pd.DataFrame(ablation1_data), palette="magma")
    plt.title(f"Ablation 1: Impact of GNN Embeddings ({dataset_name.upper()})", fontweight='bold')
    plt.ylabel(metric_filter)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ablation1_gnn_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Ablation 2
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Configuration", y="Score", data=pd.DataFrame(ablation2_data), palette="mako")
    plt.title(f"Ablation 2: Impact of Causal Batch Constraints ({dataset_name.upper()})", fontweight='bold')
    plt.ylabel(metric_filter)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"ablation2_bcq_{dataset_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved Ablation plots for {dataset_name.upper()}")

if __name__ == "__main__":
    plot_ablation_study("obd")
    plot_ablation_study("kuairec")
