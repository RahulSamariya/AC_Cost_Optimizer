from enum import Enum
import numpy as np

class OccupancyMode(Enum):
    FIXED = "fixed"
    OFFICE = "office"
    VARIABLE = "variable"

def occupancy_level(step: int, mode: OccupancyMode = OccupancyMode.FIXED, dt_min: int = 10) -> float:
    """
    Returns the occupancy level (0.0 to 1.0) for a given time step.
    
    Args:
        step: Current discrete time step.
        mode: The occupancy model to use.
        dt_min: Time step duration in minutes (default 10).
        
    Returns:
        float: Occupancy level (normalized 0.0 to 1.0).
    """
    minutes_per_day = 1440
    current_minutes = (step * dt_min) % minutes_per_day
    current_hour = current_minutes / 60.0

    if mode == OccupancyMode.FIXED:
        return 0.5  # Consistent 50% occupancy
    
    elif mode == OccupancyMode.OFFICE:
        # Typical office hours: 09:00 - 17:00
        # High occupancy (0.8 - 1.0) during work hours, low (0.05) otherwise.
        if 9.0 <= current_hour < 17.0:
            return 0.9
        elif (8.0 <= current_hour < 9.0) or (17.0 <= current_hour < 18.0):
            return 0.3  # Arrival/Departure transitions
        else:
            return 0.05
            
    elif mode == OccupancyMode.VARIABLE:
        # Stochastic/Variable: base + sinusoidal + random jumps
        import random
        base = 0.4
        variation = 0.3 * np.sin(2 * np.pi * (current_minutes - 480) / minutes_per_day)
        
        # Sudden occupancy changes (random 5% chance)
        jump = random.uniform(-0.3, 0.3) if random.random() < 0.05 else 0.0
        
        return float(np.clip(base + variation + jump, 0.0, 1.0))
        
    return 0.0
