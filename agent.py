from functools import total_ordering
from os import truncate
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime
from memory import ReplayBuffer
from model import *

class Agent:

    def __init__(self, env : gym.Env, model_count=5):
        
        self.max_memory_size = 10000
        self.batch_size = 512
        
        self.env = env

        obs, info = env.reset()

        action = env.action_space.sample() 

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.memory = ReplayBuffer(max_size=self.max_memory_size,
                                   input_shape=obs.shape,
                                   n_actions=action.shape[0],
                                   input_device=self.device,
                                   output_device=self.device) 

        self.model = EnsembleModel(obs_shape=obs.shape[0], action_shape=action.shape[0],
                                   device=self.device, hidden_dim=256)



    def plan_action(self, current_state, horizon=10, num_samples=100):
        current_state = torch.tensor(current_state, dtype=torch.float32).to(self.device)

        action_sequences = torch.rand(num_samples, horizon, 2).to(self.device) * 2 - 1

        states = current_state.unsqueeze(0).expand(num_samples, -1)
        total_returns = torch.zeros(num_samples, device=self.device)

        for t in range(horizon):

            actions = action_sequences[:, t, :]
            delta_states, rewards, delta_uncertainty, reward_uncertainty = self.model.predict(states, actions)
            states = states + delta_states

            uncertainty_penalty = delta_uncertainty.sum(-1) + reward_uncertainty.squeeze(-1)

            total_returns += rewards.squeeze(-1)
            total_returns -= 25.0 * uncertainty_penalty

        best_idx = torch.argmax(total_returns)

        return action_sequences[best_idx, 0].cpu().numpy()


    def test(self):
        done = False
        episode_reward = 0.0
        obs, info = self.env.reset()

        # self.model.load_the_model("latest.pt")
        self.model.load_the_model("best.pt")

        while not done:
            action = self.plan_action(current_state=obs)

            obs, reward, done, truncated, info = self.env.step(action)
            episode_reward = episode_reward + float(reward)
            self.env.render()

            if(done or truncated):
                break

        print(f"Test episode finished. Reward: {episode_reward}")


    def train(self, episodes):

        total_steps = 0
        best_score = -1000

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        writer = SummaryWriter(summary_writer_name)

        for episode in range(episodes):

            done = False
            episode_reward = 0.0

            obs, info = self.env.reset()

            while not done:

                if episode < 10:
                    action = self.env.action_space.sample()
                else:
                    action = self.plan_action(current_state=obs)

                next_obs, reward, done, truncated, info = self.env.step(action)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward = episode_reward + float(reward)

                if(done or truncated):
                    break

            if(episode_reward > best_score):
                best_score = episode_reward
                self.model.save_the_model('best.pt')

            self.model.save_the_model('latest.pt')

            writer.add_scalar("Score/Episode Reward", episode_reward, episode)
            print(f"Completed episode {episode} with score {episode_reward}")

            if(episode % 10 == 0):
                for _ in range(100):
                    if(self.memory.can_sample(batch_size=self.batch_size)):
                        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size=self.batch_size)

                        rewards = rewards.unsqueeze(1)
                        dones = dones.unsqueeze(1).float()

                        loss = self.model.train_step(states=states,
                                                     next_states=next_states,
                                                     actions=actions,
                                                     rewards=rewards)

                        writer.add_scalar("Loss/model", loss, total_steps)

                        total_steps += 1

                        




