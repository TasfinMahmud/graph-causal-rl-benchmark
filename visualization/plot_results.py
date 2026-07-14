import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_obd():
    print("Plotting OBD Tournament Results...")
    df = pd.read_csv(r"../data/tournament_obd_metrics.csv")
    df = df[df['Metric'] == 'SNIPW_CTR']
    df = df.sort_values(by="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Score", data=df, palette="viridis")
    
    plt.axhline(y=0.00495, color='gray', linestyle='--', alpha=0.8, label='Random Baseline (0.00495)')
    # plt.title("OBD Dataset: OPE (SNIPW) Evaluation", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("SNIPW CTR Estimator", fontsize=14)
    plt.xlabel("RL Models", fontsize=14)
    plt.legend()
    
    # Add values on top of bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.5f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
                   
    plt.tight_layout()
    # Save to data dir
    plt.savefig(r"../results/obd_results.png", dpi=300)
    # Save to IEEE paper dir
    plt.savefig(r"../results/obd_snipw.png", dpi=300)
    plt.close()

def plot_kuairec():
    print("Plotting KuaiRec Tournament Results...")
    df = pd.read_csv(r"../data/tournament_kuairec_metrics.csv")
    df = df.sort_values(by="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Model", y="Score", data=df, palette="viridis")
    
    # plt.title("KuaiRec Dataset: Exact Ground-Truth Evaluation", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Hit Rate (Ground Truth)", fontsize=14)
    plt.xlabel("RL Models", fontsize=14)
    
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.5f'), 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'center', 
                   xytext = (0, 9), 
                   textcoords = 'offset points')
                   
    plt.tight_layout()
    # Save to data dir
    plt.savefig(r"../results/kuairec_results.png", dpi=300)
    # Save to IEEE paper dir
    plt.savefig(r"../results/kuairec_exact.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    os.makedirs(r"../data", exist_ok=True)
    plot_obd()
    plot_kuairec()
    print("Graphs successfully generated and saved to data/ directory!")
