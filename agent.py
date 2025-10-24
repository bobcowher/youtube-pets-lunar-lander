from functools import total_ordering
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
                                   device=self.device)



    def plan_action(self):
        pass

    def test(self):
        pass

    def train(self, episodes):

        total_steps = 0
        best_score = -1000

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        writer = SummaryWriter(summary_writer_name)

        for episode in range(episodes):

            done = False
            episode_reward = 0

            obs, info = self.env.reset()

            while not done:

                if episode < 10:
                    action = self.env.action_space.sample()
                else:
                    action = self.env.action_space.sample()
                    # TODO: Pull from plan action

                next_obs, reward, done, truncated, info = self.env.step(action)

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward = episode_reward + reward

                if(done or truncated):
                    break

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

                        




