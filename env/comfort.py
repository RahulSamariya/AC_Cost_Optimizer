import numpy as np
from .constants import ADAPTIVE_BASE, ADAPTIVE_SLOPE, SIGMA_COMFORT

def adaptive_target(T_out: float) -> float:
    """
    Calculates the adaptive comfort temperature target.
    """
    target = ADAPTIVE_SLOPE * T_out + ADAPTIVE_BASE
    return float(np.clip(target, 20.0, 26.0))

def comfort_score(T_in: float, T_out: float, action: float) -> float:
    """
    Calculates a comfort score based on the deviation from the adaptive target.
    """
    target = adaptive_target(T_out)
    deviation = abs(T_in - target)
    
    # Penalize deviations using Gaussian-like penalty
    score = np.exp(-0.5 * (deviation / SIGMA_COMFORT)**2)
    
    # Small penalty for drafts/noise (action == 3 is turbo)
    action_penalty = 0.05 if action >= 0.8 else 0.0
    
    return float(np.clip(score - action_penalty, 0.0, 1.0))
