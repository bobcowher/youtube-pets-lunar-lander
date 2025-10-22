import torch
import os

class ReplayBuffer:

    def __init__(self, 
                 max_size,
                 input_shape,
                 n_actions,
                 input_device,
                 output_device='cpu'):
        
        self.mem_size = max_size
        self.mem_ctr  = 0

        override = os.getenv("REPLAY_BUFFER_MEMORY")

        if override in ["cpu", "cuda:0", "cuda:1"]:
            print("Received replay buffer memory override.")
            self.input_device = override
        else:
            self.input_device = input_device

        self.output_device = output_device

        self.state_memory = torch.zeros(
            (max_size, *input_shape), dtype=torch.float32, device=self.input_device
        )
        
        self.next_state_memory = torch.zeros(
            (max_size, *input_shape), dtype=torch.float32, device=self.input_device
        )

        self.action_memory = torch.zeros(
            (self.mem_size, n_actions), dtype=torch.float32, device=self.input_device
        )

        self.reward_memory = torch.zeros(max_size, dtype=torch.float32, device=self.input_device)
        
        self.terminal_memory = torch.zeros(max_size, dtype=torch.bool, device=self.input_device)

        print(f"Replay buffer created with input device {self.input_device} and output device {self.output_device}")


    def can_sample(self, batch_size):
        return self.mem_ctr >= batch_size * 10


    def store_transition(self, state, action, reward, next_state, done):

        idx = self.mem_ctr % self.mem_size

        self.state_memory[idx] = torch.as_tensor(state, 
                                                 dtype=torch.float32, 
                                                 device=self.input_device)
        
        self.next_state_memory[idx] = torch.as_tensor(next_state, 
                                                      dtype=torch.float32, 
                                                      device=self.input_device)

        self.action_memory[idx] = torch.as_tensor(action, dtype=torch.float32, device=self.input_device)

        self.reward_memory[idx] = float(reward)
        self.terminal_memory[idx] = bool(done)

        self.mem_ctr += 1


    def sample_buffer(self, batch_size):

        max_mem = min(self.mem_ctr, self.mem_size)
        batch = torch.randint(0, max_mem, (batch_size,),
                              device=self.input_device, dtype=torch.int64)

        states = self.state_memory[batch].to(self.output_device, dtype=torch.float32)
        next_states = self.next_state_memory[batch].to(self.output_device, dtype=torch.float32)

        rewards = self.reward_memory[batch].to(self.output_device)
        dones = self.terminal_memory[batch].to(self.output_device)
        actions = self.action_memory[batch].to(self.output_device)

        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)

        return states, actions, rewards, next_states, dones


    def print_mem_size(self):
        max_mem = min(self.mem_ctr, self.mem_size)
        print(f"Currently {max_mem} records loaded in the buffer")

















        





