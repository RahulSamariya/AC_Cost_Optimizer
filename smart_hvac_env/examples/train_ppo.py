import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
import numpy as np

# Import our custom environment
from smart_hvac_env.env import SmartHVACEnv

class HVACMetricsCallback(BaseCallback):
    """
    Custom callback for logging HVAC-specific metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.metrics = []

    def _on_step(self) -> bool:
        # Extract info from the environment
        info = self.locals['infos'][0]
        self.logger.record('hvac/pmv', info.get('pmv', 0))
        self.logger.record('hvac/ppd', info.get('ppd', 0))
        self.logger.record('hvac/energy_cost', info.get('energy_cost', 0))
        self.logger.record('hvac/power_w', info.get('hvac_power', 0))
        return True

def train():
    # 1. Create Environment
    # Difficulty curriculum can be implemented by wrapping or switching envs
    env = SmartHVACEnv(difficulty='easy', scenario='office')
    
    # 2. Initialize PPO Agent
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01, # Encourage exploration
        verbose=1,
        tensorboard_log="./hvac_ppo_logs/"
    )

    # 3. Training with Curriculum (Simulated)
    print("Training on EASY difficulty...")
    model.learn(total_timesteps=100000, callback=HVACMetricsCallback())
    
    print("Switching to MEDIUM difficulty...")
    env.difficulty = 'medium'
    model.learn(total_timesteps=200000, callback=HVACMetricsCallback())
    
    print("Switching to HARD difficulty...")
    env.difficulty = 'hard'
    model.learn(total_timesteps=300000, callback=HVACMetricsCallback())

    # 4. Save model
    model.save("hvac_ppo_agent")

if __name__ == "__main__":
    train()
