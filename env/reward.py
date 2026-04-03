from .state import ThermalState
from .pricing import electricity_price, PricingMode
from .comfort import adaptive_target
from .occupancy import occupancy_level
from .constants import LAMBDA_COST, W_COMFORT

def compute_reward(
    state: ThermalState, 
    action: int, 
    comfort: float, 
    pricing_mode: PricingMode = PricingMode.FLAT
) -> float:
    """
    Computes the scalar reward for the current time step.
    
    Formula: reward = (comfort * W_COMFORT) - (real_cost * LAMBDA_COST) - discomfort_penalty
    """
    # 1. Calculate REAL currency cost ($)
    price = electricity_price(state.time_step, mode=pricing_mode)
    real_cost = state.energy_consumed * price
    
    # 2. Comfort reward
    comfort_term = comfort * W_COMFORT
    
    # 3. Cost penalty (using tunable lambda)
    # Why LAMBDA_COST? It defines how many "comfort points" 1 dollar is worth.
    # At 0.2, the agent is willing to sacrifice 1% comfort to save 5 cents.
    cost_penalty = real_cost * LAMBDA_COST
    
    # 4. Discomfort penalty (scaled by occupancy)
    # If no one is in the room, discomfort matters much less.
    occ = occupancy_level(state.time_step)
    discomfort_penalty = (1.0 - comfort) * occ
    
    # Final balanced reward
    reward = comfort_term - cost_penalty - discomfort_penalty
    
    return float(reward)
