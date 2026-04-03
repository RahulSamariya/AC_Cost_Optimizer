from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from cool_budget_ai.tasks.task_configs import make_env, TASK_REGISTRY
from cool_budget_ai.env.thermal_env import ThermalEnv, StepResult

app = FastAPI()

# Global environment instance (default to hard)
env = make_env("hard")

class ResetRequest(BaseModel):
    task: Optional[str] = "hard"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action: int

@app.post("/reset")
async def reset(request: ResetRequest = Body(...)):
    global env
    if request.task.lower() in TASK_REGISTRY:
        env = make_env(request.task.lower())
    else:
        raise HTTPException(status_code=400, detail=f"Invalid task: {request.task}")
    
    # reset returns StepResult
    result = env.reset(seed=request.seed)
    # We return the whole StepResult as JSON
    return result

@app.post("/step")
async def step(request: StepRequest = Body(...)):
    # step returns StepResult
    result = env.step(request.action)
    return result

@app.get("/state")
async def get_state():
    return env.state()

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Default port for app.py is 8000 in this file, 
    # but Docker uses 7860. Uvicorn will respect port arg.
    uvicorn.run(app, host="0.0.0.0", port=8000)
