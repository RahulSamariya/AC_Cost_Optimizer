import logging
import threading
from typing import Optional
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.state_manager import state_manager
from tasks.generator import generate_tickets
from models.action import Action
from models.state import State
from graders.grader import grade_episode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OpenEnv Multi-Agent Triage Server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_event():
    logger.info("OpenEnv Ticket Triage Server ready on port 7860")

@app.post("/reset")
async def reset(episode_id: str = Body(...), difficulty: Optional[str] = Body(None)):
    # Create tickets based on difficulty
    num_tickets = 5 if difficulty == "easy" else 10
    tickets = generate_tickets(num_tickets, difficulty)
    
    env = state_manager.create_env(episode_id, tickets)
    return env.reset()

@app.post("/step")
async def step(episode_id: str = Body(...), action: Action = Body(...)):
    env = state_manager.get_env(episode_id)
    return env.step(action)

@app.get("/state")
async def get_state(episode_id: str):
    env = state_manager.get_env(episode_id)
    return State(
        current_index=env.current_index,
        tickets=env.tickets,
        total_reward=env.total_reward,
        step_count=env.step_count,
        done=env.done,
        episode_id=env.episode_id
    )

@app.get("/trajectory")
async def get_trajectory(episode_id: str):
    env = state_manager.get_env(episode_id)
    return {
        "trajectory": env.trajectory,
        "report": grade_episode(env.trajectory)
    }
