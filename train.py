from agent import Agent
import gymnasium as gym
import time

start_time = time.perf_counter()

episodes = 1200

env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5,
               render_mode="rgb_array")

agent = Agent(env=env)

agent.train(episodes=episodes)

end_time = time.perf_counter()
elapsed_time = end_time - start_time

print(f"Elapsed time was : {elapsed_time}")

