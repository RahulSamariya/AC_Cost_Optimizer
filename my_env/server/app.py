from fastapi import FastAPI, HTTPException
from .my_env_environment import SupportTicketEnv
from ..models import Action, Observation, Ticket, Priority, TicketCategory

app = FastAPI(title="Support Ticket RL Environment API")

# Default Knowledge Base Data
KB_DATA = {
    "KB_001": "Technical reset instructions.",
    "KB_002": "Billing refund cycle details.",
    "KB_003": "Feature request process."
}

# Initial Tasks
DEFAULT_TICKETS = [
    Ticket(id="T1", description="Billing issue.", category=TicketCategory.BILLING, priority=Priority.LOW, required_kb_id="KB_002"),
    Ticket(id="T2", description="Technical issue.", category=TicketCategory.TECHNICAL, priority=Priority.HIGH, required_kb_id="KB_001"),
]

# Global environment instance
env = SupportTicketEnv(DEFAULT_TICKETS, KB_DATA)

@app.post("/reset", response_model=Observation)
async def reset():
    """Resets the environment and returns the initial observation."""
    return env.reset()

@app.post("/step")
async def step(action: Action):
    """Executes a single step in the environment given an action."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="Environment must be reset first.")
    
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": info
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}
