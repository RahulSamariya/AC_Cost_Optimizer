import numpy as np
import math

def compute_pmv_ppd(t_air: float, t_radiant: float, v_air: float, humidity_pct: float, clo: float, met: float) -> dict:
    """
    Computes Predicted Mean Vote (PMV) and Predicted Percentage Dissatisfied (PPD)
    based on ASHRAE-55 Fanger model.
    """
    # 1. Constants for Fanger model
    # Metabolic rate in W/m2 (1 met = 58.15 W/m2)
    M = met * 58.15
    # External work is usually 0
    W = 0
    # Thermal insulation of clothing in m2K/W
    icl = 0.155 * clo
    
    # 2. Convert humidity to water vapor pressure (kPa)
    # Using Antoine equation for saturation pressure and multiplying by relative humidity
    p_a = (humidity_pct / 100.0) * 0.1 * math.exp(18.956 - 4030.18 / (t_air + 235))

    # 3. Intermediate variables
    fcl = 1.0 + 1.29 * icl if clo <= 0.5 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(v_air) # Forced convection coefficient
    
    # 4. Iteratively solve for clothing surface temperature (tcl)
    tcl = t_air + (35.5 - t_air) / (3.5 * (6.45 * icl + 0.1)) # Initial guess
    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100
    p4 = p1 * t_air
    p5 = 308.7 - 0.028 * M + p2 * ((t_radiant + 273) / 100)**4
    
    xn = tcl / 100
    xf = xn
    n = 0
    eps = 0.0001 # Tolerance for convergence
    
    while n < 150:
        xf = (xf + xn) / 2
        # Heat transfer coefficient
        hcn = 2.38 * abs(100 * xf - 273.15 - t_air)**0.25
        hc = hcf if hcf > hcn else hcn
        
        xn = (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc)
        if abs(xn - xf) < eps:
            break
        n += 1
    
    tcl = 100 * xn - 273.15
    
    # 5. Calculate PMV components
    # Heat loss through skin
    hl1 = 3.05 * 10**-3 * (5733 - 6.99 * (M - W) - p_a * 1000)
    # Heat loss through sweating
    hl2 = 0.42 * (M - W - 58.15) if (M - W) > 58.15 else 0
    # Latent respiration heat loss
    hl3 = 1.7 * 10**-5 * M * (5867 - p_a * 1000)
    # Dry respiration heat loss
    hl4 = 0.0014 * M * (34 - t_air)
    # Radiation heat loss
    hl5 = 3.96 * fcl * (xn**4 - (t_radiant / 100 + 2.7315)**4)
    # Convection heat loss
    hl6 = fcl * hc * (tcl - t_air)
    
    # Thermal sensation coefficient
    ts = 0.303 * math.exp(-0.036 * M) + 0.028
    
    # Calculate PMV
    pmv = ts * (M - W - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    pmv = np.clip(pmv, -3.0, 3.0)
    
    # Calculate PPD
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    ppd = np.clip(ppd, 5.0, 100.0)
    
    return {"pmv": float(pmv), "ppd": float(ppd)}

# Mapping functions for env use
def get_v_air(fan_speed_idx: int) -> float:
    return {0: 0.0, 1: 0.1, 2: 0.25, 3: 0.5, 4: 1.2}.get(fan_speed_idx, 0.1)

def get_met(occupancy_idx: int) -> float:
    return {0: 0.8, 1: 1.2, 2: 2.0}.get(occupancy_idx, 1.2)

if __name__ == "__main__":
    # Test case from Section 02
    res = compute_pmv_ppd(t_air=25, t_radiant=26, v_air=0.1, humidity_pct=50, clo=0.5, met=1.2)
    print(f"Test PMV/PPD: {res}")
