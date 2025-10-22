import gymnasium as gym
import torch
# from torch.utils.tensorboard import SummaryWriter
import datetime
from memory import ReplayBuffer

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

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs
                episode_reward = episode_reward + reward

                if(done or truncated):
                    break

            self.memory.print_mem_size()



