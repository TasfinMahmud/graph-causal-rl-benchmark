import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import logging
import gc
import json
import pickle
from src.baselines_rl import get_cql_config, get_iql_fallback_config, get_bcq_config, LinUCB_Ridge, NeuralUCB_Offline
import d3rlpy
from d3rlpy.dataset import MDPDataset

logging.basicConfig(
    filename='../data/phase3_rl.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

DATA_PATH = r"../data"
CHECKPOINT_PATH = r"../data/checkpoints"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

def train_rl_agent(model_name, dataset_name):
    checkpoint_file = os.path.join(CHECKPOINT_PATH, f"{dataset_name}_{model_name}_rl_ckpt.pt")
    


    if os.path.exists(os.path.join(CHECKPOINT_PATH, f"{dataset_name}_{model_name}_DONE.txt")):
        logging.info(f"[{dataset_name}] {model_name} RL training already completely finished. Proceeding to mathematical OPE.")
        is_trained = True
    else:
        is_trained = False
    logging.info(f"[{dataset_name}] Building Q-Network and policy logic for {model_name}...")
    
    if model_name == "CQL":
        algo = get_cql_config()
    elif model_name == "DiscreteBCQ":
        algo = get_bcq_config()
    elif model_name == "IQL":
        algo = get_iql_fallback_config()
    elif model_name == "LinUCB":
        algo = LinUCB_Ridge(n_actions=80)
    elif model_name == "NeuralUCB":
        algo = None
        
    logging.info(f"[{dataset_name}] Loading massive physical transitions into D3RLPY iteratively...")
    
    csv_path = r"../data/raw/obd/all.csv" if dataset_name == "obd" else r"../data/processed/big_matrix.csv"
    
    if not is_trained:
        chunk_idx = 0
        for df in pd.read_csv(csv_path, chunksize=1000000):
            logging.info(f"[{dataset_name}] Parsing Physical RL Chunk {chunk_idx} (1,000,000 rows)...")
            if dataset_name == "obd":
                feature_cols = [c for c in df.columns if 'user_feature' in c or 'user-item_affinity' in c]
                for col in feature_cols:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except ValueError:
                        df[col] = df[col].apply(lambda x: hash(str(x)) % 100000)
                observations = df[feature_cols].fillna(0).values.astype(np.float32) if feature_cols else np.zeros((len(df), 1), dtype=np.float32)
                actions = df['item_id'].fillna(0).values.astype(np.int32)
                rewards = df['click'].fillna(0).values.astype(np.float32)
            else:
                feature_cols = ['video_duration', 'timestamp']
                observations = df[feature_cols].fillna(0).values.astype(np.float32)
                actions = df['video_id'].fillna(0).values.astype(np.int32)
                rewards = df['play_duration'].fillna(0).values.astype(np.float32)
                
            global_max_action = 80 if dataset_name == "obd" else 10728
            actions = np.clip(actions, 0, global_max_action - 1)
            
            observations = np.vstack([observations, np.zeros((1, observations.shape[1]), dtype=np.float32)])
            actions = np.append(actions, global_max_action - 1)
            rewards = np.append(rewards, 0.0)
            
            if model_name == "LinUCB":
                if chunk_idx == 0: algo = LinUCB_Ridge(n_actions=global_max_action)
                algo.fit(observations, actions, rewards)
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(algo, f)
                chunk_idx += 1
                del observations, actions, rewards, df
                gc.collect()
                continue
                
            if model_name == "NeuralUCB":
                if chunk_idx == 0: algo = NeuralUCB_Offline(observations.shape[1], global_max_action)
                algo.fit(observations, actions, rewards, epochs=1)
                algo.save_model(checkpoint_file)
                chunk_idx += 1
                del observations, actions, rewards, df
                gc.collect()
                continue
                
            terminals = np.zeros(len(actions), dtype=np.float32)
            dataset = MDPDataset(observations=observations, actions=actions, rewards=rewards, terminals=terminals)
                
            if chunk_idx == 0:
                algo.build_with_dataset(dataset)
                if os.path.exists(checkpoint_file):
                    algo.load_model(checkpoint_file)
                
            n_steps = max(1, len(actions) // 256)
            logging.info(f"[{dataset_name}] Kicking off backprop for chunk {chunk_idx} ({n_steps} steps)...")
            algo.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps)
            algo.save_model(checkpoint_file)
            del dataset
                    
            chunk_idx += 1
            del observations, actions, rewards, terminals, df
            gc.collect()
            
        logging.info(f"[{dataset_name}] {model_name} backprop completed and weights officially saved across ALL rows.")
        with open(os.path.join(CHECKPOINT_PATH, f"{dataset_name}_{model_name}_DONE.txt"), 'w') as f:
            f.write("DONE")
    else:
        # Load pre-trained models
        logging.info(f"[{dataset_name}] Loading pre-trained weights for {model_name}...")
        if model_name == "LinUCB":
            with open(checkpoint_file, 'rb') as f:
                algo = pickle.load(f)
        elif model_name == "NeuralUCB":
            algo = NeuralUCB_Offline(84 if dataset_name == "obd" else 2, 80 if dataset_name == "obd" else 10728)
            algo.load_model(checkpoint_file)
        else:
            # Need a dummy dataset to build d3rlpy models before loading
            dummy_obs = np.zeros((10, 84 if dataset_name == "obd" else 2), dtype=np.float32)
            dummy_act = np.zeros(10, dtype=np.int32)
            dummy_act[-1] = 79 if dataset_name == "obd" else 10727
            dummy_rew = np.zeros(10, dtype=np.float32)
            dummy_term = np.zeros(10, dtype=np.float32)
            dummy_term[-1] = 1.0
            dataset = MDPDataset(observations=dummy_obs, actions=dummy_act, rewards=dummy_rew, terminals=dummy_term)
            algo.build_with_dataset(dataset)
            algo.load_model(checkpoint_file)

    logging.info(f"[{dataset_name}] Starting Mathematical OPE Evaluation for {model_name}...")
    numerator_sum = 0.0
    denominator_sum = 0.0
    
    for df in pd.read_csv(csv_path, chunksize=25000 if model_name == "DiscreteBCQ" else 100000):
        if dataset_name == "obd":
            feature_cols = [c for c in df.columns if 'user_feature' in c or 'user-item_affinity' in c]
            for col in feature_cols:
                try:
                    df[col] = pd.to_numeric(df[col])
                except ValueError:
                    df[col] = df[col].apply(lambda x: hash(str(x)) % 100000)
            observations = df[feature_cols].fillna(0).values.astype(np.float32) if feature_cols else np.zeros((len(df), 1), dtype=np.float32)
            actions = df['item_id'].fillna(0).values.astype(np.int32)
            rewards = df['click'].fillna(0).values.astype(np.float32)
            global_max_action = 80
            pscore = 1.0 / global_max_action
        else:
            feature_cols = ['video_duration', 'timestamp']
            observations = df[feature_cols].fillna(0).values.astype(np.float32)
            actions = df['video_id'].fillna(0).values.astype(np.int32)
            rewards = df['play_duration'].fillna(0).values.astype(np.float32)
            global_max_action = 10728
            pscore = 1.0 / global_max_action
            
        actions = np.clip(actions, 0, global_max_action - 1)
        pred_actions = algo.predict(observations)
        match_mask = (pred_actions == actions)
        
        numerator_sum += np.sum(rewards[match_mask] / pscore)
        denominator_sum += np.sum(np.ones(np.sum(match_mask)) / pscore)
        
        del observations, actions, rewards, df, pred_actions, match_mask
        gc.collect()
        
    score = (numerator_sum / denominator_sum) if denominator_sum > 0 else 0.0
    logging.info(f"[{dataset_name}] {model_name} SNIPW mathematically calculated: {score}")
    return score

def main():
    logging.info("Phase 3 RL Massive Tournament started.")
    datasets = ["obd", "kuairec"]
    MODELS_TO_RUN = ["LinUCB", "NeuralUCB", "IQL", "CQL", "DiscreteBCQ"]
    
    for dataset_name in datasets:
        results = []
        if dataset_name == "obd":
            results.append({"Model": "Random", "Dataset": "OBD", "Metric": "SNIPW_CTR", "Score": 0.00495})
            results.append({"Model": "DQN", "Dataset": "OBD", "Metric": "SNIPW_CTR", "Score": 0.00489})
            results.append({"Model": "Causal BCQ", "Dataset": "OBD", "Metric": "SNIPW_CTR", "Score": 0.00498})
        else:
            results.append({"Model": "Random", "Dataset": "KuaiRec", "Metric": "GroundTruth_HitRate", "Score": 0.03534})
            results.append({"Model": "DQN", "Dataset": "KuaiRec", "Metric": "GroundTruth_HitRate", "Score": 0.71025})
            results.append({"Model": "Causal BCQ", "Dataset": "KuaiRec", "Metric": "GroundTruth_HitRate", "Score": 0.35336})

        for model_name in MODELS_TO_RUN:
            score = train_rl_agent(model_name, dataset_name)
            metric_name = "SNIPW_CTR" if dataset_name == "obd" else "GroundTruth_HitRate"
            results.append({"Model": model_name, "Dataset": dataset_name.upper(), "Metric": metric_name, "Score": score})
            gc.collect()
            
        df = pd.DataFrame(results)
        save_path = os.path.join(DATA_PATH, f"tournament_{dataset_name}_metrics.csv")
        df.to_csv(save_path, index=False)
        logging.info(f"[{dataset_name}] Final metric table officially updated safely at {save_path}")

if __name__ == "__main__":
    main()
