# ── Thermal dynamics ─────────────────────────────────────────
# Easy Mode Constants
ALPHA_OUT_EASY  = 0.01   
BETA_LOAD_EASY  = 0.05   
GAMMA_COOL_EASY = 0.1    

# Hard Mode Constants (Increased gain and coupling)
ALPHA_OUT_HARD  = 0.02   # Faster coupling to ambient
BETA_LOAD_HARD  = 0.08   # Higher internal heat gain
GAMMA_COOL_HARD = 0.08   # Slightly less efficient cooling

DT_MIN      = 10.0   
TEMP_CLIP   = (15.0, 45.0)  

# ── AC modes ─────────────────────────────────────────────────
POWER_KW          = {0: 0.00, 1: 0.05, 2: 0.80, 3: 2.20}
COOLING_INTENSITY = {0: 0.00, 1: 0.00, 2: 0.40, 3: 1.00}

# ── Comfort model ─────────────────────────────────────────────
SIGMA_COMFORT     = 1.5    
ADAPTIVE_BASE     = 17.8
ADAPTIVE_SLOPE    = 0.31

# ── Reward & Grading ──────────────────────────────────────────
LAMBDA_COST       = 0.2    # Penalty for real currency cost
W_COMFORT         = 1.0    
W_ENERGY          = 1.0    

# ── Episode ───────────────────────────────────────────────────
EPISODE_STEPS = 144   
