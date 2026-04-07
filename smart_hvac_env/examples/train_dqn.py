import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np

# Import our custom environment
from smart_hvac_env.env import SmartHVACEnv

def train():
    # DQN is highly effective for discrete action spaces like our 14-choice set.
    # We focus on the 'server_room' scenario where internal heat is constant.
    env = SmartHVACEnv(difficulty='hard', scenario='server_room')
    
    # Custom wrapper for anti-oscillation
    class AntiOscillationWrapper(gym.ActionWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.last_action = None

        def action(self, action):
            # Penalize if action changes (this is actually handled in our reward.py already)
            # but we can add extra logic here if needed.
            self.last_action = action
            return action

    env = AntiOscillationWrapper(env)

    # 2. Initialize DQN Agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=5000,
        batch_size=128,
        tau=1.0, # Hard update
        target_update_interval=1000,
        exploration_fraction=0.3, # Explore heavily for 30% of steps
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./hvac_dqn_logs/"
    )

    # 3. Training
    print("Training DQN on HARD Server Room...")
    model.learn(total_timesteps=300000)

    # 4. Save
    model.save("hvac_dqn_agent")

if __name__ == "__main__":
    train()
