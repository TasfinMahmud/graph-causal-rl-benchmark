import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

DATA_DIR = r"../data"
OUTPUT_DIR = r"../results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_metrics(dataset_name, y_label, metric_filter):
    csv_file = os.path.join(DATA_DIR, f"tournament_{dataset_name}_metrics.csv")
    if not os.path.exists(csv_file):
        logging.error(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    # Filter for the specific metric (e.g., SNIPW_CTR or MRDR_CTR)
    df_metric = df[df['Metric'] == metric_filter].copy()
    
    if df_metric.empty:
        logging.warning(f"No data found for metric {metric_filter} in {dataset_name}.")
        return

    # Sort models to have BCQ and Contextual Bandits grouped nicely
    # Make sure BCQ/DiscreteBCQ/Causal BCQ are highlighted
    # We will color Core vs Baseline
    def get_model_type(model):
        if "BCQ" in model:
            return "Core (Offline RL)"
        elif "UCB" in model:
            return "Baseline (Bandit)"
        elif "QL" in model:
            return "Baseline (Offline RL)"
        else:
            return "Other"

    df_metric['Type'] = df_metric['Model'].apply(get_model_type)
    df_metric = df_metric.sort_values(by='Score', ascending=False)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.figure(figsize=(10, 6))
    
    # Barplot
    ax = sns.barplot(x="Model", y="Score", hue="Type", data=df_metric, dodge=False, palette="viridis")
    
    plt.title(f"Performance Comparison: {metric_filter} ({dataset_name.upper()})", fontweight='bold')
    plt.ylabel(y_label)
    plt.xlabel("")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, f"barplot_{dataset_name}_{metric_filter}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved barplot to {save_path}")

if __name__ == "__main__":
    # OBD Metrics
    plot_metrics("obd", "Click-Through Rate (CTR)", "SNIPW_CTR")
    plot_metrics("obd", "MRDR Evaluation Score", "MRDR_CTR")
    
    # KuaiRec Metrics
    plot_metrics("kuairec", "Hit Rate @ 10", "GroundTruth_HitRate")
    plot_metrics("kuairec", "MRDR Hit Rate Score", "MRDR_HitRate")
