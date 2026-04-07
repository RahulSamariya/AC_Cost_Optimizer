# Support Ticket Routing & Resolution Environment

A production-grade Reinforcement Learning environment compatible with OpenEnv.

## Reward Function
The dense reward $R$ is defined as:
$$R = \mathbb{1}_{res} \cdot 5.0 + \mathbb{1}_{class} \cdot 1.0 - \mathbb{1}_{bad\_esc} \cdot 2.0 - 0.5 \cdot t$$

## Project Structure
- `models.py`: Pydantic V2 schemas.
- `env.py`: The MDP transition logic.
- `tasks.py`: Dataset definitions for benchmarks.
- `graders.py`: QoS-based scoring.

## How to Run
```bash
python run_all.py
```
