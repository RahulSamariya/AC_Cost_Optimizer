from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import asyncio

from .env import SmartHVACEnv
from .replay_buffer import AsyncReplayBuffer

app = FastAPI(title="SmartHVACEnv API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
env = SmartHVACEnv()
buffer = AsyncReplayBuffer()

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"
    scenario: Optional[str] = "room"

class StepRequest(BaseModel):
    action: int

@app.post("/reset")
async def reset(request: ResetRequest = Body(...)):
    global env
    env = SmartHVACEnv(difficulty=request.difficulty, scenario=request.scenario)
    obs, info = env.reset()
    return {"observation": obs.tolist(), "info": info}

@app.post("/step")
async def step(request: StepRequest = Body(...)):
    if not (0 <= request.action <= 13):
        raise HTTPException(status_code=400, detail="Action must be between 0 and 13")
    
    # Store state before step
    state_before = env._get_obs()
    
    # Take step
    obs, reward, terminated, truncated, info = env.step(request.action)
    
    # Log to replay buffer
    metadata = {
        "pmv": info.get('pmv'),
        "ppd": info.get('ppd'),
        "hvac_power": info.get('hvac_power'),
        "energy_cost": info.get('energy_cost'),
        "reward": reward
    }
    await buffer.add(state_before, request.action, reward, obs, terminated or truncated, metadata)
    
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info
    }

@app.get("/buffer/stats")
async def get_buffer_stats():
    return buffer.get_statistics()

@app.post("/buffer/save")
async def save_buffer(request: Dict[str, str]):
    filepath = request.get("filepath", "buffer.npz")
    await buffer.save(filepath)
    return {"status": "saved", "path": filepath}

@app.post("/buffer/load")
async def load_buffer(request: Dict[str, str]):
    filepath = request.get("filepath", "buffer.npz")
    await buffer.load(filepath)
    return {"status": "loaded", "path": filepath}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
