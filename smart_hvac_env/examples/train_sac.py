import gymnasium as gym
from stable_baselines3 import SAC
import numpy as np

# Import our custom environment
from smart_hvac_env.env import SmartHVACEnv

# Note: SAC by default works on continuous actions. 
# For Discrete(14), we can use a wrapper or the SB3 MultiInputPolicy / MlpPolicy
# SB3's SAC supports discrete actions via internal Softmax if the action space is discrete.

def train():
    # 1. Create Environment
    env = SmartHVACEnv(difficulty='medium', scenario='room')
    
    # Reward Scaling is critical for SAC entropy tuning.
    # We wrap the env to divide rewards by 2.0 to bring them closer to [-1, 1].
    class RewardScaleWrapper(gym.RewardWrapper):
        def reward(self, reward):
            return reward / 2.0

    env = RewardScaleWrapper(env)

    # 2. Initialize SAC Agent
    # We use a large buffer_size for off-policy training.
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto", # Automatically tune entropy temperature
        verbose=1,
        tensorboard_log="./hvac_sac_logs/"
    )

    # 3. Training
    print("Training SAC on MEDIUM scenario...")
    model.learn(total_timesteps=200000)

    # 4. Save
    model.save("hvac_sac_agent")

if __name__ == "__main__":
    train()
