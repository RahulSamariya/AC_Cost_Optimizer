import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from typing import Dict, Any, Tuple, Optional

from .thermal import ThermalModel
from .comfort import compute_pmv_ppd, get_v_air, get_met
from .reward import compute_reward
from .pricing import PricingSchedule

class SmartHVACEnv(gym.Env):
    """
    Main Gymnasium Environment for Smart HVAC control.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, difficulty='easy', scenario='room'):
        super().__init__()
        self.difficulty = difficulty
        self.scenario = scenario
        
        # 1. Action Space: Discrete(14)
        self.action_space = spaces.Discrete(14)
        
        # 2. Observation Space: Box(15,)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(15,), dtype=np.float32)
        
        # 3. Component Initializations
        self.thermal = ThermalModel()
        self.pricing = PricingSchedule(mode='tou' if difficulty != 'easy' else 'flat')
        
        # 4. State Variables
        self.t_air = 22.0
        self.t_mass = 22.0
        self.t_outdoor = 25.0
        self.humidity = 50.0
        self.occupancy = 1
        self.fan_speed = 1
        self.setpoint = 23.0
        self.turbo = False
        self.eco = False
        self.ventilation_level = 0
        self.window_open = False
        self.solar_gain = 0.0
        
        self.current_step = 0
        self.max_steps = 288 # 24h at 5-min intervals
        
        self.last_actions = {}
        self.current_actions = {}

    def _get_obs(self):
        """
        Returns normalized observation vector [0, 1].
        """
        price = self.pricing.get_price(self.current_step * 5 / 60)
        next_price = self.pricing.get_next_hour_price(self.current_step * 5 / 60)
        price_delta = next_price - price
        
        # ASHRAE parameters for PMV
        v_air = get_v_air(self.fan_speed)
        met = get_met(self.occupancy)
        # Assuming t_radiant approx t_mass
        comfort_dict = compute_pmv_ppd(self.t_air, self.t_mass, v_air, self.humidity, 0.5, met)
        pmv = comfort_dict['pmv']
        ppd = comfort_dict['ppd']
        
        peak_soon = 1.0 if self.pricing.is_peak_soon(self.current_step * 5 / 60) else 0.0
        
        # Normalization logic
        obs = np.array([
            (self.t_air - 15) / 25,             # 0: indoor_temp [15, 40]
            (self.t_outdoor + 10) / 55,         # 1: outdoor_temp [-10, 45]
            (self.humidity - 10) / 80,          # 2: humidity [10, 90]
            self.occupancy / 2,                 # 3: occupancy {0, 1, 2}
            self.fan_speed / 4,                 # 4: fan_speed {0-4}
            (price - 0.05) / 0.45,              # 5: current_price [0.05, 0.50]
            (next_price - 0.05) / 0.45,         # 6: next_hour_price [0.05, 0.50]
            (price_delta + 0.45) / 0.90,        # 7: price_delta [-0.45, 0.45]
            (pmv + 3) / 6,                      # 8: pmv [-3, 3]
            ppd / 100,                          # 9: ppd [5, 100]
            (self.t_mass - 15) / 25,            # 10: mass_temp [15, 40]
            self.solar_gain / 500,              # 11: solar [0, 500]
            self.ventilation_level / 2,         # 12: vent_state {0, 1, 2}
            float(self.window_open),            # 13: window_state {0, 1}
            peak_soon                           # 14: peak_soon {0, 1}
        ], dtype=np.float32)
        
        return np.clip(obs, 0, 1)

    def _get_info(self):
        v_air = get_v_air(self.fan_speed)
        met = get_met(self.occupancy)
        comfort_dict = compute_pmv_ppd(self.t_air, self.t_mass, v_air, self.humidity, 0.5, met)
        return {
            "pmv": comfort_dict['pmv'],
            "ppd": comfort_dict['ppd'],
            "t_air": self.t_air,
            "t_mass": self.t_mass,
            "energy_price": self.pricing.get_price(self.current_step * 5 / 60),
            "step": self.current_step
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Difficulty-based randomization
        if self.difficulty == 'easy':
            self.t_outdoor = random.uniform(18, 28)
            self.solar_gain = random.uniform(0, 100)
        elif self.difficulty == 'medium':
            self.t_outdoor = random.uniform(10, 35)
            self.solar_gain = random.uniform(0, 300)
        else: # hard
            self.t_outdoor = random.uniform(-5, 42)
            self.solar_gain = random.uniform(0, 500)
            
        self.t_air = random.uniform(20, 28)
        self.t_mass = self.t_air
        self.current_step = 0
        self.pricing.reset_with_jitter()
        
        # Initial internal actions
        self.setpoint = 23.0
        self.fan_speed = 1
        self.turbo = False
        self.eco = False
        self.ventilation_level = 0
        self.window_open = False
        
        self.current_actions = {
            'setpoint': self.setpoint, 'fan_speed': self.fan_speed, 
            'turbo': self.turbo, 'eco': self.eco, 
            'ventilation_level': self.ventilation_level, 'window_open': self.window_open,
            'occupancy': self.occupancy
        }
        self.last_actions = self.current_actions.copy()
        
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.last_actions = self.current_actions.copy()
        
        # --- Apply Discrete Actions ---
        if action == 1: self.setpoint = max(16, self.setpoint - 1)
        elif action == 2: self.setpoint = min(30, self.setpoint + 1)
        elif action == 3: self.fan_speed = 1
        elif action == 4: self.fan_speed = 2
        elif action == 5: self.fan_speed = 3
        elif action == 6: 
            self.turbo = True
            self.eco = False
            self.fan_speed = 4
        elif action == 7: 
            self.turbo = False
            self.fan_speed = 1
        elif action == 8: 
            self.eco = True
            self.turbo = False
        elif action == 9: self.eco = False
        elif action == 10: self.window_open = True
        elif action == 11: self.window_open = False
        elif action == 12: self.ventilation_level = min(2, self.ventilation_level + 1)
        elif action == 13: self.ventilation_level = max(0, self.ventilation_level - 1)
        
        # --- Update Environment ---
        # Random walk for humidity and outdoor temp (small changes)
        self.humidity = np.clip(self.humidity + random.uniform(-1, 1), 10, 90)
        
        # Server Room Scenario overrides
        internal_load = 100
        if self.scenario == 'server_room':
            self.occupancy = 0
            internal_load = 5000
        else:
            # Random occupancy changes
            if random.random() < 0.02:
                self.occupancy = random.randint(0, 2)
        
        self.current_actions = {
            'setpoint': self.setpoint, 'fan_speed': self.fan_speed, 
            'turbo': self.turbo, 'eco': self.eco, 
            'ventilation_level': self.ventilation_level, 'window_open': self.window_open,
            'occupancy': self.occupancy, 'internal_load': internal_load,
            'is_sunny': self.solar_gain > 200
        }
        
        # Thermal Step
        self.t_air, self.t_mass, hvac_power = self.thermal.step(
            self.t_air, self.t_mass, self.t_outdoor, self.current_actions
        )
        
        # --- Compute Rewards ---
        v_air = get_v_air(self.fan_speed)
        met = get_met(self.occupancy)
        comfort_dict = compute_pmv_ppd(self.t_air, self.t_mass, v_air, self.humidity, 0.5, met)
        
        price = self.pricing.get_price(self.current_step * 5 / 60)
        next_price = self.pricing.get_next_hour_price(self.current_step * 5 / 60)
        
        reward, reward_info = compute_reward(
            comfort_dict['pmv'], comfort_dict['ppd'], hvac_power, self.fan_speed,
            self.turbo, self.eco, self.ventilation_level, self.window_open,
            self.t_outdoor, self.t_air, price, next_price, self.last_actions, self.current_actions
        )
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        info = self._get_info()
        info.update(reward_info)
        info['hvac_power'] = hvac_power
        info['energy_cost'] = (hvac_power / 1000) * (5/60) * price # Cost for this 5-min step
        
        return obs, float(reward), terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            hour = (self.current_step * 5 // 60) % 24
            minute = (self.current_step * 5) % 60
            print(f"[{hour:02d}:{minute:02d}] T_air: {self.t_air:.1f}C | PMV: {self._get_info()['pmv']:.2f} | "
                  f"Power: {self._get_info().get('hvac_power', 0):.0f}W | Reward: {self._get_info().get('comfort_reward', 0):.2f}")
