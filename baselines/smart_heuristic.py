import numpy as np
from ..env.state import ThermalState
from ..env.comfort import adaptive_target

class SmartHeuristicAgent:
    """
    A price and occupancy aware heuristic agent.
    Adjusts target temperature to balance cost and comfort.
    """
    def __init__(self):
        pass

    def select_action(self, state: ThermalState, info: dict) -> int:
        """
        Selects a discrete action based on a dynamic target.
        """
        # [outside_temp, load, fan, cpu_temp, cost_this_step, time_step]
        outside_temp = state.ambient_temp
        current_temp = state.cpu_temp
        
        # Info contains current price and occupancy
        price = info.get('price', 0.12)
        occupancy = info.get('occupancy', 0.5)
        
        # 1. Start with the adaptive comfort target
        base_target = adaptive_target(outside_temp)
        
        # 2. Adjust target based on price
        # If price > 0.18, allow it to be warmer to save money
        price_offset = 0.0
        if price > 0.18:
            price_offset = 2.0  # Let it get hot
        elif price < 0.08:
            price_offset = -1.0 # Pre-cool!
            
        # 3. Adjust target based on occupancy
        # If no one is there, we don't care about comfort as much
        occ_offset = 0.0
        if occupancy < 0.1:
            occ_offset = 3.0  # Very relaxed
        elif occupancy < 0.4:
            occ_offset = 1.0  # Slightly relaxed
            
        target = base_target + price_offset + occ_offset
        # Hard limits
        target = np.clip(target, 21.0, 28.0)
        
        # 4. Control Logic (Hysteresis-like)
        deviation = current_temp - target
        
        if deviation > 1.5:
            return 3  # Turbo
        elif deviation > 0.0:
            return 2  # Eco
        elif deviation > -1.0:
            return 1  # Fan
        else:
            return 0  # Off
