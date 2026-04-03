import numpy as np
from ..env.state import ThermalState
from ..env.comfort import adaptive_target

class RuleBasedAgent:
    """
    A simple rule-based agent for thermal control.
    It uses a target temperature and adjusts cooling based on the deviation.
    """
    def __init__(self, target_temp: float = 23.0, sensitivity: float = 0.2):
        self.target_temp = target_temp
        self.sensitivity = sensitivity

    def select_action(self, state: ThermalState, step: int) -> int:
        """
        Selects a discrete cooling action based on current temperature deviation.
        
        Args:
            state: Current ThermalState.
            step: Current time step.
            
        Returns:
            int: Action index {0: OFF, 1: FAN, 2: ECO, 3: TURBO}.
        """
        current_target = self.target_temp
        deviation = state.cpu_temp - current_target
        
        if deviation > 2.0:
            return 3  # Turbo
        elif deviation > 0.5:
            return 2  # Eco
        elif deviation > -0.5:
            return 1  # Fan only
        else:
            return 0  # Off
