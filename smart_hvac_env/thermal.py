import numpy as np

class ThermalModel:
    """
    Two-node RC thermal model for a room environment.
    Node 1: Indoor air temperature (T_air)
    Node 2: Thermal mass temperature (T_mass)
    """
    def __init__(self, 
                 C_air=1000 * 1.2 * 50,      # J/K (Air heat capacity, 50 m3 room)
                 C_mass=500 * 800 * 200,     # J/K (Concrete thermal mass, 200 kg equiv.)
                 R_mass_air=0.05,            # K/W Thermal resistance mass <-> air
                 window_solar_gain=200,      # W solar gain when sunny
                 occupant_heat=80,           # W per occupant
                 dt=300):                    # Seconds (5 min timestep)
        
        self.C_air = C_air
        self.C_mass = C_mass
        self.R_mass_air = R_mass_air
        self.window_solar_gain = window_solar_gain
        self.occupant_heat = occupant_heat
        self.dt = dt
        
        # Physical constants
        self.rho_air = 1.225 # kg/m3
        self.cp_air = 1005   # J/kg.K

    def calculate_hvac_power(self, t_air, t_setpoint, mode_dict):
        """
        Simple HVAC power model with ECO and TURBO modes.
        """
        # Base cooling power using a proportional controller
        # If t_air > t_setpoint, we need cooling (negative power)
        error = t_air - t_setpoint
        kp = 500 # W/K
        base_power = -kp * error if error > 0 else 0
        
        # Mode multipliers
        eco_mult = 1.3 if mode_dict.get('eco', False) else 1.0
        turbo_power_mult = 2.0 if mode_dict.get('turbo', False) else 1.0
        
        # Actual power consumed (W)
        hvac_w = base_power * turbo_power_mult
        
        # Effective cooling provided to the air (W)
        # ECO mode provides more cooling per Watt consumed
        q_hvac = hvac_w * eco_mult
        
        return hvac_w, q_hvac

    def step(self, t_air, t_mass, t_outdoor, action_dict):
        """
        Performs one Euler integration step (5 minutes).
        Returns: t_air_new, t_mass_new, hvac_power_w
        """
        # 1. HVAC Power
        hvac_power_w, q_hvac = self.calculate_hvac_power(t_air, action_dict['setpoint'], action_dict)
        
        # 2. Internal Gains
        q_internal = action_dict['occupancy'] * self.occupant_heat + action_dict.get('internal_load', 100)
        
        # 3. Solar Gain
        q_solar = self.window_solar_gain if action_dict.get('is_sunny', False) else 0
        
        # 4. Ventilation
        # v_rate mapping based on level: 0=0.01, 1=0.05, 2=0.15 m3/s
        v_rates = [0.01, 0.05, 0.15]
        v_rate = v_rates[action_dict.get('ventilation_level', 0)]
        q_vent = v_rate * self.rho_air * self.cp_air * (t_outdoor - t_air)
        
        # 5. Mass-Air Heat Exchange
        q_mass_air = (t_mass - t_air) / self.R_mass_air
        
        # 6. Euler Integration for Air Node
        # dT_air/dt = (1/C_air) * [Q_hvac + Q_solar + Q_vent + Q_mass_air + Q_internal]
        dt_air = (self.dt / self.C_air) * (q_hvac + q_solar + q_vent + q_mass_air + q_internal)
        t_air_new = t_air + dt_air
        
        # 7. Euler Integration for Mass Node
        # dT_mass/dt = (1/C_mass) * [-Q_mass_air]
        dt_mass = (self.dt / self.C_mass) * (-q_mass_air)
        t_mass_new = t_mass + dt_mass
        
        return float(t_air_new), float(t_mass_new), float(abs(hvac_power_w))
