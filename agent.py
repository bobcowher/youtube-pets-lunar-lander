import gymnasium as gym
import torch
# from torch.utils.tensorboard import SummaryWriter
import datetime

class Agent:

    def __init__(self, env : gym.Env, model_count=5):
        
        self.max_memory_size = 10000
        self.batch_size = 512
        
        self.env = env



    def plan_action(self):
        pass

    def test(self):
        pass

    def train(self, episodes):

        total_steps = 0
        best_score = -1000

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

                # TODO: Store memory transitions. 

                obs = next_obs
                episode_reward = episode_reward + reward

                if(done or truncated):
                    break



