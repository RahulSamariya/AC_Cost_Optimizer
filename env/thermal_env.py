import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional

from .state import ThermalState
from .dynamics import step_thermal
from .weather import outdoor_temp
from .pricing import electricity_price, PricingMode
from .occupancy import occupancy_level, OccupancyMode
from .comfort import comfort_score
from .reward import compute_reward

from .constants import (
    POWER_KW, COOLING_INTENSITY, DT_MIN, EPISODE_STEPS,
    ALPHA_OUT_EASY, ALPHA_OUT_HARD, 
    BETA_LOAD_EASY, BETA_LOAD_HARD,
    GAMMA_COOL_EASY, GAMMA_COOL_HARD
)

@dataclass
class StepResult:
    observation: List[float]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    state: Dict[str, Any]

class ThermalEnv:
    """
    Pure Python ThermalEnv (Standalone, No Gymnasium).
    """
    def __init__(self, max_steps: int = EPISODE_STEPS, dt_min: int = DT_MIN, 
                 pricing_mode: PricingMode = PricingMode.FLAT,
                 occupancy_mode: OccupancyMode = OccupancyMode.OFFICE,
                 is_hard: bool = False):
        self.max_steps = max_steps
        self.dt_min = dt_min
        self.pricing_mode = pricing_mode
        self.occupancy_mode = occupancy_mode
        self.is_hard = is_hard

        # Select dynamics based on difficulty
        self.alpha_out = ALPHA_OUT_HARD if is_hard else ALPHA_OUT_EASY
        self.beta_load = BETA_LOAD_HARD if is_hard else BETA_LOAD_EASY
        self.gamma_cool = GAMMA_COOL_HARD if is_hard else GAMMA_COOL_EASY

        self.state_obj: Optional[ThermalState] = None
        self.current_step = 0
        self.total_cost = 0.0
        self.last_comfort = 1.0

    def _get_obs(self) -> List[float]:
        # [outside_temp, load, fan, cpu_temp, energy, time_step]
        if self.state_obj is None:
            return [0.0] * 6
        return self.state_obj.to_array().tolist()

    def state(self) -> Dict[str, Any]:
        if self.state_obj is None:
            return {}
        return asdict(self.state_obj)

    def reset(self, seed: Optional[int] = None) -> StepResult:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.current_step = 0
        self.total_cost = 0.0
        
        # Add variability to initial temperature
        initial_temp = 25.0 + random.uniform(-3.0, 3.0)
        
        # Pre-compute time-varying signals (with randomness)
        self._outdoor_temps = [outdoor_temp(s, dt_min=self.dt_min) + random.uniform(-1, 1) 
                               for s in range(self.max_steps + 1)]
        self._prices = [electricity_price(s, mode=self.pricing_mode, dt_min=self.dt_min) 
                        for s in range(self.max_steps + 1)]
        self._occupancies = [occupancy_level(s, mode=self.occupancy_mode, dt_min=self.dt_min) 
                             for s in range(self.max_steps + 1)]
        
        self.state_obj = ThermalState(
            ambient_temp=self._outdoor_temps[0],
            server_load=self._occupancies[0] * 0.8,
            fan_speed=0.5,
            cpu_temp=initial_temp,
            energy_consumed=0.0,
            time_step=self.current_step
        )
        
        self.last_comfort = 1.0
        
        return StepResult(
            observation=self._get_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"comfort": self.last_comfort, "cost": 0.0, "total_cost": 0.0, "energy": 0.0},
            state=self.state()
        )

    def step(self, action: int) -> StepResult:
        intensity = COOLING_INTENSITY[int(action)]
        power_kw = POWER_KW[int(action)]
        
        # 1. Thermal Physics (difficulty-aware)
        dt = self.dt_min
        new_cpu_temp = self.state_obj.cpu_temp + (self.state_obj.server_load * self.beta_load * dt) - (intensity * self.gamma_cool * dt)
        new_cpu_temp += self.alpha_out * (self.state_obj.ambient_temp - self.state_obj.cpu_temp) * dt
        
        # 2. Update context
        self.current_step += 1
        amb_temp = self._outdoor_temps[self.current_step]
        occ = self._occupancies[self.current_step]
        price = self._prices[self.current_step]
        
        # 3. Consumption and Cost
        step_energy = power_kw * (self.dt_min / 60.0)
        step_cost = step_energy * price
        self.total_cost += step_cost
        
        # 4. Update State
        self.state_obj.ambient_temp = amb_temp
        self.state_obj.server_load = occ * 0.8 + 0.1
        self.state_obj.cpu_temp = new_cpu_temp
        self.state_obj.energy_consumed = step_energy
        self.state_obj.time_step = self.current_step
        
        # 5. Reward
        self.last_comfort = comfort_score(new_cpu_temp, amb_temp, intensity)
        reward = compute_reward(self.state_obj, int(action), self.last_comfort, pricing_mode=self.pricing_mode)
        
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            "comfort": self.last_comfort,
            "cost": step_cost,
            "total_cost": self.total_cost,
            "energy": step_energy
        }
        
        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            state=self.state()
        )
