import pytest
import numpy as np
from cool_budget_ai.env.thermal_env import ThermalEnv, StepResult
from cool_budget_ai.env.constants import EPISODE_STEPS

def test_env_reset():
    """Confirms that reset() initializes the state correctly and returns StepResult."""
    env = ThermalEnv()
    res = env.reset(seed=42)
    assert isinstance(res, StepResult)
    assert isinstance(res.observation, list)
    assert len(res.observation) == 6
    assert env.current_step == 0
    assert res.reward == 0.0
    assert res.terminated == False
    assert res.state != {}

def test_env_step():
    """Confirms that step() processes actions and returns StepResult."""
    env = ThermalEnv()
    env.reset()
    res = env.step(2) # Eco mode
    assert env.current_step == 1
    assert isinstance(res.reward, float)
    assert hasattr(res, 'observation')
    assert hasattr(res, 'reward')
    assert hasattr(res, 'terminated')
    assert hasattr(res, 'truncated')
    assert hasattr(res, 'info')
    assert not res.terminated

def test_episode_length():
    """Confirms the environment terminates after 144 steps."""
    env = ThermalEnv()
    env.reset()
    # 144 steps total. We take 143 steps.
    for _ in range(EPISODE_STEPS - 1):
        env.step(0)
    # The 144th step should terminate.
    res = env.step(0)
    assert res.terminated == True

def test_state_method():
    """Confirms the state() method returns a non-empty dict after reset."""
    env = ThermalEnv()
    env.reset()
    s = env.state()
    assert isinstance(s, dict)
    assert "cpu_temp" in s
    assert "time_step" in s
