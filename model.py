import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsModel(nn.Module):

    def __init__(self, hidden_dim=256, obs_shape=None, action_shape=None):
        super(DynamicsModel, self).__init__()

        self.fc1 = nn.Linear(obs_shape + action_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.obs_diff_output = nn.Linear(hidden_dim, obs_shape)
        self.reward_output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        obs_diff = self.obs_diff_output(x)
        reward = self.reward_output(x)

        return obs_diff, reward


class EnsembleModel:

    def __init__(self, num_models=5, hidden_dim=256, obs_shape=None, action_shape=None,
                 device=None, learning_rate=0.0001):
        
        self.models = [DynamicsModel(hidden_dim=hidden_dim, obs_shape=obs_shape,
                                     action_shape=action_shape).to(device) for _ in range(num_models)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in self.models]
        self.model_save_dir = 'models'


    def train_step(self, states, actions, next_states, rewards):
        total_loss = 0

        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):

            indices = torch.randint(0, len(states), (len(states),))
            batch_states = states[indices]
            batch_next_states = next_states[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]

            obs_diffs = batch_next_states - batch_states

            predicted_obs_diffs, predicted_rewards = model(batch_states, batch_actions)

            loss = F.mse_loss(torch.cat([predicted_obs_diffs, predicted_rewards], dim=-1),
                              torch.cat([obs_diffs, batch_rewards], dim=-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.models)

    
    def predict(self, states, actions):

        with torch.no_grad():
            obs_diffs = []
            rewards = []
        
            for model in self.models:
                obs_diff, reward = model(states, actions)
                obs_diffs.append(obs_diff)
                rewards.append(reward)

            obs_diffs = obs_diffs.stack(obs_diffs)
            rewards = torch.stack(rewards)

            obs_uncertainty = obs_diffs.var(0)
            reward_uncertainty = rewards.var(0)

            avg_obs_diff = obs_diffs.mean(0)
            avg_reward = rewards.mean(0)

            return avg_obs_diff, avg_reward, obs_uncertainty, reward_uncertainty

    
    def save_the_model(self, filename='latest'):
        os.makedirs(self.model_save_dir, exist_ok=True)

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f'{self.model_save_dir}/{filename}_{i}.pt')

    def load_the_model(self, filename='latest'):
        for i, model in enumerate(self.models):
            file_path = f"{self.model_save_dir}/{filename}_{i}.pt"

            try:
                model.load_state_dict(torch.load(file_path))
                print(f"Loaded weights from {file_path}")
            except FileNotFoundError:
                print(f"No weights file found at {file_path}")






