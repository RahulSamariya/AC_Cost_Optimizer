# SmartHVACEnv Training Guide

Welcome to the Expert RL Training Guide for HVAC optimization.

## 1. Algorithm Selection
- **PPO (Proximal Policy Optimization):** Best for general stability. Use it as your baseline. It's on-policy, meaning it's less efficient with data but more stable.
- **SAC (Soft Actor-Critic):** Best for data efficiency and peak performance. It's off-policy and has maximum entropy, allowing for robust exploration.
- **DQN (Deep Q-Network):** Excellent for discrete actions. It's the simplest and often very effective for the `server_room` scenario.
- **Offline RL:** Use this if you have a large dataset of manual logs (via the API) and cannot risk training a live agent on expensive HVAC hardware.

## 2. Reward Hacking Prevention
- **Turbo Abuse:** If the agent keeps cycling Turbo on/off to save energy while cooling fast, increase the `instability_penalty` in `reward.py`.
- **Window Oscillation:** If the agent opens and closes windows repeatedly, add a higher `window_penalty` or an action change penalty.
- **Setpoint Cycling:** RL agents may try to cycle the setpoint by +/-1C to gain small comfort bonuses. Use a `switching_cost` to penalize this.

## 3. Anti-Oscillation Tricks
- **Action Repeat Penalty:** Penalize the agent if its action choice changes more than twice in an hour.
- **Observation History:** Feed the last 3 observations (Frame Stacking) into the agent so it knows the "velocity" of temperature change.
- **Action Smoothing:** Use a low-level PID controller for setpoint changes while the RL agent only chooses the target.

## 4. Key Metrics to Track
Monitor these in TensorBoard to see if your agent is truly learning:
- `pmv_mean`: Should stay within [-0.5, +0.5].
- `ppd_mean`: Should stay below 10%.
- `comfort_pct`: % of steps within the comfort zone.
- `energy_kwh_per_episode`: Should decrease over training.
- `action_change_rate`: Lower is better for equipment health.

## 5. Curriculum Learning
Start training on **EASY** mode. Once the agent reaches a mean reward of 1.5, switch to **MEDIUM**. Finally, move to **HARD** to teach the agent how to handle extreme weather spikes and volatile energy prices.

## 6. Failure Modes & Fixes
- **Agent stays in comfort but uses too much power:** Increase `hvac_penalty`.
- **Agent saves energy but discomfort is high:** Increase `pmv_reward` weights.
- **Reward stays at 0:** Check your normalization. All observations MUST be between 0 and 1.
- **Agent doesn't pre-cool before peak hours:** Ensure the agent sees the `peak_price_soon` observation and that the `pre_cooling_bonus` is high enough.
