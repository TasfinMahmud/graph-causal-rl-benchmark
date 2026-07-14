import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import d3rlpy
from d3rlpy.algos import DiscreteCQLConfig, DiscreteSACConfig, DiscreteBCQConfig
from d3rlpy.preprocessing import StandardObservationScaler, StandardRewardScaler
from sklearn.linear_model import Ridge

def get_bcq_config():
    """
    Batch-Constrained deep Q-learning (BCQ).
    The core methodology of the thesis.
    """
    return DiscreteBCQConfig(
        batch_size=256,
        learning_rate=1e-3,
        action_flexibility=0.3
    ).create(device='cuda:0')

def get_cql_config():
    """
    Conservative Q-Learning (CQL) baseline.
    Penalizes out-of-distribution actions directly in the Q-function.
    """
    return DiscreteCQLConfig(
        batch_size=256,
        learning_rate=1e-3,
        alpha=5.0
    ).create(device='cuda:0')

def get_iql_fallback_config():
    """
    Fallback for IQL using Discrete SAC if IQL is continuous-only in this d3rlpy version.
    Provides an entropy-regularized robust baseline.
    """
    return DiscreteSACConfig(
        batch_size=256,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3
    ).create(device='cuda:0')

class LinUCB_Ridge:
    """
    Linear Upper Confidence Bound (LinUCB) Offline approximation using Ridge Regression.
    Acts as a deliberately weak linear baseline to prove the necessity of Deep RL.
    """
    def __init__(self, n_actions, alpha=1.0):
        self.n_actions = n_actions
        self.models = [Ridge(alpha=alpha) for _ in range(n_actions)]
        self.is_fitted = [False] * n_actions

    def fit(self, observations, actions, rewards):
        for a in range(self.n_actions):
            mask = (actions == a)
            if np.sum(mask) > 0:
                self.models[a].fit(observations[mask], rewards[mask])
                self.is_fitted[a] = True

    def predict(self, observations):
        n_samples = observations.shape[0]
        q_values = np.zeros((n_samples, self.n_actions))
        for a in range(self.n_actions):
            if self.is_fitted[a]:
                q_values[:, a] = self.models[a].predict(observations)
            else:
                q_values[:, a] = -1e9 # Penalize unfitted actions
        
        return np.argmax(q_values, axis=1)

class NeuralUCB_Offline:
    """
    Offline Neural Contextual Bandit baseline inspired by NeuralUCB (Zhou et al. 2020).
    Implements a Deep PyTorch Multi-Layer Perceptron predicting the rewards for actions.
    This effectively gives the baseline the exact representation power of NeuralUCB, 
    but for offline pre-collected datasets without online covariance inversions.
    """
    def __init__(self, obs_dim, n_actions, device='cuda:0'):
        self.n_actions = n_actions
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()

    def fit(self, observations, actions, rewards):
        self.model.train()
        
        obs_t = torch.tensor(observations, dtype=torch.float32).to(self.device)
        act_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        rew_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)

        batch_size = 5000
        n_samples = obs_t.size(0)
        
        for i in range(0, n_samples, batch_size):
            b_obs = obs_t[i:i+batch_size]
            b_act = act_t[i:i+batch_size]
            b_rew = rew_t[i:i+batch_size]
            
            self.optimizer.zero_grad()
            q_values = self.model(b_obs)
            q_selected = q_values.gather(1, b_act)
            
            loss = self.criterion(q_selected, b_rew)
            loss.backward()
            self.optimizer.step()

    def predict(self, observations):
        self.model.eval()
        obs_t = torch.tensor(observations, dtype=torch.float32).to(self.device)
        
        pred_actions = []
        batch_size = 5000
        n_samples = obs_t.size(0)
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                b_obs = obs_t[i:i+batch_size]
                q_values = self.model(b_obs)
                b_pred = q_values.argmax(dim=1).cpu().numpy()
                pred_actions.append(b_pred)
                
        return np.concatenate(pred_actions)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
