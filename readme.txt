AC COST OPTIMIZER (COOL BUDGET AI)
================================

An HVAC control environment for benchmarking thermal comfort and energy efficiency.

1. INSTALLATION
---------------
Prerequisites: Python 3.8+

Step 1: Create a virtual environment
    python -m venv .venv

Step 2: Activate the environment
    Windows: .venv\Scripts\activate
    Linux/macOS: source .venv/bin/activate

Step 3: Install dependencies
    pip install numpy gymnasium pytest fastapi uvicorn pydantic requests
    pip install -e .

Note: The code uses the 'cool_budget_ai' namespace for imports.


2. RUNNING THE REST API
-----------------------
Start the FastAPI server to interact with the environment via HTTP:

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints:
- POST /reset : Reset env (e.g., {"task": "easy"})
- POST /step  : Take action (e.g., {"action": 1})
- GET /state  : Get current state
- GET /health : Check status


3. RUNNING EVALUATION
---------------------
Run a baseline agent evaluation on a specific task:

    python run_eval.py --task easy --render


4. RUNNING TESTS
----------------
Verify the environment logic using pytest:

    pytest tests/


5. DOCKER
---------
Build and run using the provided Dockerfile:

    docker build -t ac-cost-optimizer .
    docker run -p 7860:7860 ac-cost-optimizer


6. ENVIRONMENT DETAILS
----------------------
Observation Space: [ambient_temp, server_load, fan_speed, cpu_temp, energy_consumed, time_step]
Action Space: 0:OFF, 1:FAN, 2:ECO, 3:TURBO
Tasks: easy, medium, hard
