from .state import ThermalState

def step_thermal(state: ThermalState, action: float, dt_min: float = 10.0, 
                 alpha_out: float = 0.01, beta_load: float = 0.05, gamma_cool: float = 0.1):
    """
    Simulates thermal dynamics for one time step with configurable coefficients.
    """
    T_indoor = state.cpu_temp + (state.server_load * beta_load * dt_min) - (action * gamma_cool * dt_min)
    T_indoor += alpha_out * (state.ambient_temp - state.cpu_temp) * dt_min
    
    T_wall = (T_indoor + state.ambient_temp) / 2.0
    
    return float(T_indoor), float(T_wall)
