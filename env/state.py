from dataclasses import dataclass, fields
import numpy as np

@dataclass
class ThermalState:
    ambient_temp: float
    server_load: float
    fan_speed: float
    cpu_temp: float
    energy_consumed: float
    time_step: int

    def to_array(self) -> np.ndarray:
        """Converts the state fields into a numpy array of floats."""
        return np.array([
            self.ambient_temp,
            self.server_load,
            self.fan_speed,
            self.cpu_temp,
            self.energy_consumed,
            float(self.time_step)
        ], dtype=np.float32)
