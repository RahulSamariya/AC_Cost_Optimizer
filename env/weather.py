import numpy as np

def outdoor_temp(step: int, dt_min: int = 10, mean_temp: float = 20.0, amplitude: float = 10.0) -> float:
    """
    Returns the outdoor temperature at a given time step following a sinusoidal profile.
    
    The profile is designed to have:
    - Minimum temperature at 02:00 (120 minutes from midnight).
    - Maximum temperature at 14:00 (840 minutes from midnight).
    
    Args:
        step: Current discrete time step.
        dt_min: Time step duration in minutes (default 10).
        mean_temp: Average daily temperature in Celsius.
        amplitude: Temperature variation from the mean.
        
    Returns:
        float: Calculated outdoor temperature.
    """
    minutes_per_day = 1440
    current_minutes = (step * dt_min) % minutes_per_day
    
    # Phase shift calculation:
    # Target: Min at 2:00 (120 min), Max at 14:00 (840 min)
    # A standard sine wave has min at -pi/2 and max at pi/2.
    # sin(2*pi * (t - phase_shift) / 1440)
    # At t=120, sin should be -1.
    # (120 - phase_shift) / 1440 * 2*pi = -pi/2
    # 120 - phase_shift = -360 => phase_shift = 480 (08:00)
    
    phase_shift = 480  # minutes to align peak/trough
    
    temp = mean_temp + amplitude * np.sin(2 * np.pi * (current_minutes - phase_shift) / minutes_per_day)
    return float(temp)
