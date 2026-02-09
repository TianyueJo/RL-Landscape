"""
Simple PPO implementation for the GridWorld environment.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple, Optional
from collections import deque


class PPOAgent(nn.Module):
    """PPO policy network (Actor-Critic)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64, lr: float = 3e-4):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
    
    def forward(self, obs: torch.Tensor):
        features = self.shared(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False):
        """Sample (or greedily select) an action from the policy."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_logits, value = self.forward(obs_tensor)
            dist = Categorical(logits=action_logits)
            if deterministic:
                action = torch.argmax(action_logits, dim=-1)
            else:
                action = dist.sample()
            action_logprob = dist.log_prob(action)
        return action.item(), action_logprob.item(), value.item()
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions (used for PPO updates)."""
        action_logits, values = self.forward(obs)
        dist = Categorical(logits=action_logits)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logprobs, values, entropy


class PPO:
    """PPO algorithm."""
    
    def __init__(
        self,
        env,
        obs_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 256,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cpu",
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
        self.agent = PPOAgent(obs_dim, action_dim, lr=lr).to(device)
        self.old_agent = PPOAgent(obs_dim, action_dim, lr=lr).to(device)
        self.old_agent.load_state_dict(self.agent.state_dict())
        
        # Trajectory buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset trajectory buffer."""
        self.buffer = {
            "obs": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "logprobs": [],
            "dones": [],
        }
    
    def collect_rollout(self):
        """Collect one rollout."""
        self.reset_buffer()
        obs, _ = self.env.reset()
        done = False
        
        for _ in range(self.n_steps):
            action, logprob, value = self.agent.get_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            self.buffer["obs"].append(obs)
            self.buffer["actions"].append(action)
            self.buffer["rewards"].append(reward)
            self.buffer["values"].append(value)
            self.buffer["logprobs"].append(logprob)
            self.buffer["dones"].append(done)
            
            obs = next_obs
            if done:
                obs, _ = self.env.reset()
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE (Generalized Advantage Estimation)."""
        advantages = []
        returns = []
        
        last_gae = 0
        next_value = 0  # if rollout ends without done, use 0
        
        # Compute backwards
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                last_gae = delta
            else:
                delta = rewards[step] + self.gamma * next_value - values[step]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages.insert(0, last_gae)
            returns.insert(0, last_gae + values[step])
            next_value = values[step]
        
        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)
    
    def update(self):
        """Update the policy/value networks."""
        # Compute GAE and returns
        advantages, returns = self.compute_gae(
            self.buffer["rewards"],
            self.buffer["values"],
            self.buffer["dones"],
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(self.buffer["obs"])).to(self.device)
        actions_tensor = torch.LongTensor(self.buffer["actions"]).to(self.device)
        old_logprobs_tensor = torch.FloatTensor(self.buffer["logprobs"]).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Update old agent
        self.old_agent.load_state_dict(self.agent.state_dict())
        
        # Multiple epochs of minibatch updates
        indices = np.arange(len(self.buffer["obs"]))
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_logprobs = old_logprobs_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Compute new logprobs and values
                new_logprobs, values, entropy = self.agent.evaluate_actions(batch_obs, batch_actions)
                
                # PPO clip
                ratio = torch.exp(new_logprobs - batch_old_logprobs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.functional.mse_loss(values.squeeze(), batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.value_coef * critic_loss + self.ent_coef * entropy_loss
                
                # Apply gradient update
                self.agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.agent.optimizer.step()
    
    def learn(self, total_timesteps: int):
        """Train."""
        import sys
        num_rollouts = total_timesteps // self.n_steps
        episode_returns = deque(maxlen=100)
        
        print(
            f"Starting training: {num_rollouts} rollouts, {total_timesteps} steps total...",
            flush=True,
        )
        
        total_steps = 0
        for rollout in range(num_rollouts):
            self.collect_rollout()
            total_steps += len(self.buffer["dones"])
            self.update()
            
            # Track episode returns
            episode_return = 0
            for i, done in enumerate(self.buffer["dones"]):
                episode_return += self.buffer["rewards"][i]
                if done:
                    episode_returns.append(episode_return)
                    episode_return = 0
            
            # Log every 10 rollouts or every 5000 steps
            if rollout % 10 == 0 or total_steps % 5000 == 0:
                if len(episode_returns) > 0:
                    mean_return = np.mean(episode_returns)
                    std_return = np.std(episode_returns) if len(episode_returns) > 1 else 0.0
                    min_return = np.min(episode_returns)
                    max_return = np.max(episode_returns)
                    print(f"Steps: {total_steps}/{total_timesteps} | "
                          f"Rollout: {rollout}/{num_rollouts} | "
                          f"Mean Return: {mean_return:.2f} ± {std_return:.2f} | "
                          f"Range: [{min_return:.2f}, {max_return:.2f}] | "
                          f"Episodes: {len(episode_returns)}", flush=True)
                else:
                    print(f"Steps: {total_steps}/{total_timesteps} | "
                          f"Rollout: {rollout}/{num_rollouts} | collecting data...", flush=True)
    
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Predict an action."""
        action, _, _ = self.agent.get_action(obs, deterministic=deterministic)
        return action, None

