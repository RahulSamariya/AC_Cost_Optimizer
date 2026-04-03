from ..env.thermal_env import ThermalEnv
from ..env.pricing import PricingMode
from ..env.occupancy import OccupancyMode

EASY = {
    "max_steps": 144,
    "dt_min": 10,
    "pricing_mode": PricingMode.FLAT,
    "occupancy_mode": OccupancyMode.FIXED,
    "is_hard": False
}

MEDIUM = {
    "max_steps": 144,
    "dt_min": 10,
    "pricing_mode": PricingMode.TOU,
    "occupancy_mode": OccupancyMode.OFFICE,
    "is_hard": False
}

HARD = {
    "max_steps": 144,
    "dt_min": 10,
    "pricing_mode": PricingMode.DYNAMIC,
    "occupancy_mode": OccupancyMode.VARIABLE,
    "is_hard": True
}

TASK_REGISTRY = {
    "easy": EASY,
    "medium": MEDIUM,
    "hard": HARD
}

def make_env(task_name: str = "easy") -> ThermalEnv:
    config = TASK_REGISTRY.get(task_name.lower(), EASY)
    return ThermalEnv(**config)
