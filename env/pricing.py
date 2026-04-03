from enum import Enum

class PricingMode(Enum):
    FLAT = "flat"
    TOU = "tou"        # Time-of-Use
    DYNAMIC = "dynamic"  # Real-time or stochastic

def electricity_price(step: int, mode: PricingMode = PricingMode.FLAT, dt_min: int = 10) -> float:
    """
    Returns the electricity price ($/kWh) for a given time step.
    
    Args:
        step: Current discrete time step.
        mode: The pricing model to use.
        dt_min: Time step duration in minutes (default 10).
        
    Returns:
        float: Price in currency per kWh.
    """
    minutes_per_day = 1440
    current_minutes = (step * dt_min) % minutes_per_day
    current_hour = current_minutes // 60

    if mode == PricingMode.FLAT:
        return 0.12  # Fixed 12 cents per kWh
    
    elif mode == PricingMode.TOU:
        # Peak: 14:00 - 20:00 (0.24 $/kWh)
        # Mid-Peak: 08:00 - 14:00, 20:00 - 22:00 (0.16 $/kWh)
        # Off-Peak: 22:00 - 08:00 (0.08 $/kWh)
        if 14 <= current_hour < 20:
            return 0.24
        elif (8 <= current_hour < 14) or (20 <= current_hour < 22):
            return 0.16
        else:
            return 0.08
            
    elif mode == PricingMode.DYNAMIC:
        # Dynamic: base sinusoidal variation + random spikes (2x-3x)
        import math
        import random
        base = 0.15
        variation = 0.05 * math.sin(2 * math.pi * (current_minutes - 360) / minutes_per_day)
        price = max(0.01, base + variation)
        
        # Random spikes: 5% chance of a spike
        if random.random() < 0.05:
            price *= random.uniform(2.0, 3.5)
            
        return float(price)
        
    return 0.12
