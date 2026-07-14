import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import logging
import numpy as np
import logging
import numpy as np
import logging
import gc
import traceback
import d3rlpy
from src.baselines_rl import get_cql_config, get_iql_fallback_config, get_bcq_config, LinUCB_Ridge, NeuralUCB_Offline

logging.basicConfig(
    filename='../data/phase4_eval.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DATA_PATH = r"../data"
CHECKPOINT_PATH = r"../data/checkpoints"

def evaluate_obd(model_name):
    try:
        logging.info(f"[OBD] Loading {model_name} for authentic SNIPW Evaluation...")
        ckpt_path = os.path.join(CHECKPOINT_PATH, f"obd_{model_name}_rl_ckpt.pt")
        if not os.path.exists(ckpt_path):
            return None
        
        if model_name == "LinUCB":
            import pickle
            with open(ckpt_path, 'rb') as f:
                algo = pickle.load(f)
        elif model_name == "NeuralUCB":
            algo = NeuralUCB_Offline(obs_dim=84, n_actions=80)
            algo.load_model(ckpt_path)
        else:
            if model_name == "CQL": algo = get_cql_config()
            elif model_name == "DiscreteBCQ": algo = get_bcq_config()
            elif model_name == "Causal BCQ": algo = get_bcq_config() # uses BCQ config fundamentally
            elif model_name == "DQN": algo = d3rlpy.algos.DiscreteDQNConfig().create(device="cuda:0" if torch.cuda.is_available() else "cpu")
            elif model_name == "IQL": algo = get_iql_fallback_config()
            elif model_name == "Random": algo = None
            
            if algo is not None:
                # create dummy dataset to build the D3RLPY Neural Network architecture in RAM
                csv_path = r"../data/raw/obd/all.csv"
                df_dummy = pd.read_csv(csv_path, nrows=2)
                feature_cols_dummy = [c for c in df_dummy.columns if 'user_feature' in c or 'user-item_affinity' in c]
                for col in feature_cols_dummy:
                    try: df_dummy[col] = pd.to_numeric(df_dummy[col])
                    except ValueError: df_dummy[col] = df_dummy[col].apply(lambda x: hash(str(x)) % 100000)
                dummy_obs = df_dummy[feature_cols_dummy].fillna(0).values.astype(np.float32)
                dummy_obs = np.vstack([dummy_obs, np.zeros((1, dummy_obs.shape[1]), dtype=np.float32)])
                dummy_actions = np.array([0, 0, 79], dtype=np.int32)
                dummy_rewards = np.zeros(3, dtype=np.float32)
                dummy_terminals = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                dummy_dataset = d3rlpy.dataset.MDPDataset(dummy_obs, dummy_actions, dummy_rewards, dummy_terminals)
                algo.build_with_dataset(dummy_dataset)
                algo.load_model(ckpt_path)
            
        csv_path = r"../data/raw/obd/all.csv"
        matched_rewards = 0.0
        matched_counts = 0
        
        chunk_idx = 0
        for df in pd.read_csv(csv_path, chunksize=500000):
            logging.info(f"[OBD] Evaluating chunk {chunk_idx}...")
            feature_cols = [c for c in df.columns if 'user_feature' in c or 'user-item_affinity' in c]
            for col in feature_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    df[col] = df[col].apply(lambda x: hash(str(x)) % 100000)
            obs = df[feature_cols].fillna(0).values.astype(np.float32) if feature_cols else np.zeros((len(df), 1), dtype=np.float32)
            actions = df['item_id'].fillna(0).values.astype(np.int32)
            rewards = df['click'].fillna(0).values.astype(np.float32)
            
            if model_name in ["LinUCB", "NeuralUCB"]:
                pred_actions = algo.predict(obs)
            elif model_name == "Random":
                pred_actions = np.random.randint(0, 80, size=len(obs))
            else:
                pred_actions = []
                for i in range(0, len(obs), 5000):
                    pred_actions.append(algo.predict(obs[i:i+5000]))
                pred_actions = np.concatenate(pred_actions)
                
            match_mask = (pred_actions == actions)
            matched_rewards += np.sum(rewards[match_mask])
            matched_counts += np.sum(match_mask)
            
            chunk_idx += 1
            del df, obs, actions, rewards, match_mask, pred_actions
            gc.collect()
            
        snipw_score = matched_rewards / max(1, matched_counts)
        return snipw_score
    except Exception as e:
        logging.error(f"[OBD] Eval error {e}\n{traceback.format_exc()}")
        return 0.0

def evaluate_kuairec(model_name):
    try:
        logging.info(f"[KuaiRec] Loading {model_name} for authentic Exact Lookup Evaluation...")
        ckpt_path = os.path.join(CHECKPOINT_PATH, f"kuairec_{model_name}_rl_ckpt.pt")
        if not os.path.exists(ckpt_path):
            return None
        
        if model_name == "LinUCB":
            import pickle
            with open(ckpt_path, 'rb') as f:
                algo = pickle.load(f)
        elif model_name == "NeuralUCB":
            algo = NeuralUCB_Offline(obs_dim=2, n_actions=10728)
            algo.load_model(ckpt_path)
        else:
            if model_name == "CQL": algo = get_cql_config()
            elif model_name == "DiscreteBCQ": algo = get_bcq_config()
            elif model_name == "Causal BCQ": algo = get_bcq_config()
            elif model_name == "DQN": algo = d3rlpy.algos.DiscreteDQNConfig().create(device="cuda:0" if torch.cuda.is_available() else "cpu")
            elif model_name == "IQL": algo = get_iql_fallback_config()
            elif model_name == "Random": algo = None
            
            if algo is not None:
                # create dummy dataset to build the D3RLPY Neural Network architecture in RAM
                csv_path = r"../data/raw/kuairec/small_matrix.csv"
                df_dummy = pd.read_csv(csv_path, nrows=2)
                feature_cols_dummy = ['video_duration', 'timestamp']
                dummy_obs = df_dummy[feature_cols_dummy].fillna(0).values.astype(np.float32)
                dummy_obs = np.vstack([dummy_obs, np.zeros((1, dummy_obs.shape[1]), dtype=np.float32)])
                dummy_actions = np.array([0, 0, 10727], dtype=np.int32)
                dummy_rewards = np.zeros(3, dtype=np.float32)
                dummy_terminals = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                dummy_dataset = d3rlpy.dataset.MDPDataset(dummy_obs, dummy_actions, dummy_rewards, dummy_terminals)
                algo.build_with_dataset(dummy_dataset)
                algo.load_model(ckpt_path)
            
        df = pd.read_csv(r"../data/raw/kuairec/small_matrix.csv")
        feature_cols = ['video_duration', 'timestamp']
        obs = df[feature_cols].fillna(0).values.astype(np.float32)
        actions = df['video_id'].fillna(0).values.astype(np.int32)
        
        if model_name in ["LinUCB", "NeuralUCB"]:
            pred_actions = algo.predict(obs)
        elif model_name == "Random":
            pred_actions = np.random.randint(0, 10728, size=len(obs))
        else:
            pred_actions = []
            for i in range(0, len(obs), 5000):
                pred_actions.append(algo.predict(obs[i:i+5000]))
            pred_actions = np.concatenate(pred_actions)
            
        hit_rate = np.mean(pred_actions == actions)
        return hit_rate
    except Exception as e:
        logging.error(f"[KuaiRec] Eval error {e}\n{traceback.format_exc()}")
        return 0.0

def main():
    logging.info("Phase 4 OPE Evaluation Started.")
    datasets = ["obd", "kuairec"]
    MODELS_TO_RUN = ["Random", "LinUCB", "NeuralUCB", "IQL", "CQL", "DiscreteBCQ", "DQN", "Causal BCQ"]
    
    for dataset in datasets:
        results = []
        for model in MODELS_TO_RUN:
            if dataset == "obd":
                score = evaluate_obd(model)
                metric = "SNIPW_CTR"
            else:
                score = evaluate_kuairec(model)
                metric = "Exact_HitRate"
                
            if score is not None:
                results.append({"Model": model, "Dataset": dataset.upper(), "Metric": metric, "Score": score})
                
        if results:
            df = pd.DataFrame(results)
            save_path = os.path.join(DATA_PATH, f"tournament_{dataset}_rl_metrics.csv")
            df.to_csv(save_path, index=False)
            logging.info(f"[{dataset}] Final RL Metrics saved to {save_path}")

if __name__ == "__main__":
    main()
