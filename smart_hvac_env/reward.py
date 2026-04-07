from typing import Dict, Any, Tuple

def compute_reward(pmv: float, 
                   ppd: float, 
                   hvac_power_w: float, 
                   fan_speed: int, 
                   turbo_on: bool, 
                   eco_on: bool,
                   ventilation_level: int, 
                   window_open: bool, 
                   outdoor_temp: float, 
                   indoor_temp: float,
                   current_price: float, 
                   next_price: float, 
                   last_actions: Dict[str, Any],
                   current_actions: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    Computes a composite reward balancing comfort, energy efficiency, and stability.
    """
    info = {}
    
    # --- 1. COMFORT REWARD ---
    # pmv_reward based on ASHRAE-55 comfort zones
    abs_pmv = abs(pmv)
    if abs_pmv <= 0.5:
        pmv_reward = 1.0
    elif abs_pmv <= 1.0:
        pmv_reward = 0.5
    elif abs_pmv <= 2.0:
        pmv_reward = -0.5
    else:
        pmv_reward = -1.5
    
    # ppd_bonus (negative penalty that grows with PPD)
    ppd_bonus = -0.01 * (ppd - 5.0)
    
    comfort_reward = pmv_reward + ppd_bonus
    info['pmv_reward'] = pmv_reward
    info['ppd_bonus'] = ppd_bonus
    info['comfort_reward'] = comfort_reward

    # --- 2. ENERGY PENALTY ---
    # Base hvac penalty scaled (0.001 * W)
    hvac_penalty = 0.001 * hvac_power_w
    
    # Fan speed penalty mapping
    fan_penalties = [0, 0.01, 0.03, 0.07, 0.15]
    fan_penalty = fan_penalties[fan_speed]
    
    turbo_penalty = 0.2 if turbo_on else 0.0
    vent_penalty = 0.02 * ventilation_level
    
    # Window penalty (heat loss/gain if window open and outside is hotter than inside)
    window_penalty = 0.05 if window_open and outdoor_temp > indoor_temp else 0.0
    
    # Current price multiplier (scaled by normalized TOU price)
    # Assuming current_price is in USD/kWh, base is approx 0.12
    price_mult = current_price / 0.12
    
    energy_penalty = (hvac_penalty + fan_penalty + turbo_penalty + vent_penalty + window_penalty) * price_mult
    info['energy_penalty'] = energy_penalty

    # Pre-cooling reward (bonus for cooling when it's cheap and a spike is coming)
    pre_cooling_bonus = 0.0
    if next_price > current_price * 1.3 and pmv < -0.3:
        pre_cooling_bonus = 0.3
    
    info['pre_cooling_bonus'] = pre_cooling_bonus

    # --- 3. INSTABILITY PENALTY ---
    # Penalize changing too many things at once
    num_changed = 0
    keys_to_check = ['setpoint', 'fan_speed', 'turbo', 'eco', 'ventilation_level', 'window_open']
    for key in keys_to_check:
        if current_actions.get(key) != last_actions.get(key):
            num_changed += 1
    
    instability_penalty = 0.1 * num_changed
    info['instability_penalty'] = instability_penalty

    # --- FINAL TOTAL REWARD ---
    total_reward = comfort_reward - energy_penalty - instability_penalty + pre_cooling_bonus
    
    return float(total_reward), info
