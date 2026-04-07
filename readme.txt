 1 # Support Ticket Triage: OpenEnv Live API Environment
    2
    3 A production-grade, OpenEnv-compliant Reinforcement Learning (RL) environment designed to train and evaluate AI agents in customer support ticket routing and resolution.
    4
    5 ---
    6
    7 ## 1. Project Overview
    8 This project simulates a high-volume support dashboard. It decouples the **Environment** (the world logic) from the **Agent** (the decision-maker) using a FastAPI-based REST architecture.
    9
   10 The goal for an RL agent is to maximize its cumulative reward by correctly classifying, resolving, or escalating a bank of 10 diverse support tickets while minimizing "agent fatigue" (step penalties).
   11
   12 ### Key Features
   13 - **OpenEnv Compliant**: Implements the standard `reset`, `step`, and `state` patterns via HTTP.
   14 - **Strict Decoupling**: The agent (`inference.py`) has zero access to the environment's source code or private state.
   15 - **Thread-Safe**: Uses a global mutex (`threading.Lock`) to ensure state consistency during high-concurrency API calls.
   16 - **Production Ready**: Fully containerized with Docker and managed via environment variables.
   17
   18 ---
   19
   20 ## 2. System Architecture
   21 The system follows a classic **Client-Server MDP (Markov Decision Process)** model:
  [ AGENT LAYER ]          [ NETWORK LAYER ]         [ ENVIRONMENT LAYER ]
  +--------------+         +---------------+         +-------------------+
  |              |  HTTP   |               |  Logic  |                   |
  | inference.py | <-----> |    app.py     | <-----> |   env_logic.py    |
  |   (Client)   |  JSON   |   (FastAPI)   |  Calls  |   (The Engine)    |
    1
    2 ---
    3
    4 ## 3. Getting Started
    5
    6 ### Prerequisites
    7 - Python 3.10+
    8 - OpenAI API Key (for LLM-based agents)
    9 - Docker (optional, for containerized deployment)
   10
   11 ### Installation
   12 1. **Clone the repository** and navigate to the project root.
   13 2. **Install dependencies**:
     pip install -r requirements.txt

   1 3. **Configure Environment Variables**:
   2    Copy the example environment file and fill in your keys.
     cp .env.example .env

   1    *Edit `.env` to include your `OPENAI_API_KEY`.*
   2
   3 ---
   4
   5 ## 4. Running the System
   6
   7 ### Step 1: Start the Environment (The Server)
   8 The environment must be running before the agent can interact with it.
  uvicorn app:app --host 0.0.0.0 --port 7860

   1 *The server will be live at `http://localhost:7860`.*
   2
   3 ### Step 2: Run the Agent (The Inference)
   4 In a separate terminal, execute the agent script.
  python inference.py
   1
   2 ### Step 3: Deployment with Docker
   3 To run the entire environment in an isolated container:
  docker build -t ticket-triage-env .
  docker run -p 7860:7860 --env-file .env ticket-triage-env

    1
    2 ---
    3
    4 ## 5. API Reference (OpenEnv Spec)
    5
    6 | Endpoint | Method | Request Body | Description |
    7 | :--- | :--- | :--- | :--- |
    8 | `/reset` | `POST` | N/A | Resets the environment and returns the first ticket observation. |
    9 | `/step` | `POST` | `Action` JSON | Processes an agent action and returns the next observation + reward. |
   10 | `/state` | `GET` | N/A | Returns the full internal state (for debugging and transparency). |
   11
   12 ---
   13
   14 ## 6. Reinforcement Learning Logic
   15
   16 ### Reward Matrix ($R$)
   17 The agent receives a reward $r_t$ at every step based on the following dense reward function:
   18
   19 | Action | Condition | Reward ($r_t$) | RL Signal |
   20 | :--- | :--- | :--- | :--- |
   21 | **Any** | Constant Step Penalty | `-0.5` | Encourages speed/efficiency. |
   22 | **Resolve** | `has_kb_solution == True` | `+4.5` (Net) | Reinforces correct resolution. |
   23 | **Resolve** | `has_kb_solution == False`| `-2.0` (Net) | Penalizes incorrect resolution attempts. |
   24 | **Escalate**| `priority == urgent` | `+0.5` (Net) | Reinforces safe handling of critical issues. |
   25 | **Escalate**| `priority == low` | `-2.5` (Net) | Penalizes "laziness" for simple tickets. |
   26 | **Classify**| N/A | `-0.5` (Net) | Neutral step to gather information. |
   27
   28 ### Ticket Lifecycle (Transitions)
   29 1. **PENDING**: Initial state of all tickets in the bank.
   30 2. **IN_PROGRESS**: Triggered when an agent calls `CLASSIFY`.
   31 3. **TERMINAL**: Tickets move to `RESOLVED` or `ESCALATED` states, advancing the environment pointer to the next ticket.
   32 4. **DONE**: The episode ends when the `current_index` reaches the end of the 10-ticket bank.
   33
   34 ---
   35
   36 ## 7. Security & Engineering Standards
   37 - **Secret Management**: No API keys are hardcoded. The system fails at startup if `OPENAI_API_KEY` is missing.
   38 - **Pydantic V2**: Uses the latest validation logic for high-speed JSON serialization.
   39 - **Logging**: The agent outputs a strict step-by-step log for "Experience Replay" analysis.

