import os
import requests
import json
from openai import OpenAI

# Environment Variables
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# OpenAI Client
client = OpenAI(
    base_url=f"https://api-inference.huggingface.co/v1/",
    api_key=HF_TOKEN,
)

def get_action_from_llm(obs):
    """
    Calls the LLM to decide on the next AC mode based on observations.
    """
    prompt = f"""
    You are an AI controlling a building's AC system.
    Observation: {obs}
    (Labels: ambient_temp, server_load, fan_speed, cpu_temp, energy_consumed, time_step)
    
    Choose an action:
    0: OFF
    1: FAN
    2: ECO
    3: TURBO
    
    Return ONLY the integer representing the action.
    """
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0.0
    )
    
    try:
        content = response.choices[0].message.content.strip()
        return int(content)
    except Exception:
        return 0 # Default to OFF on error

def run_episode():
    print("[START]")
    
    # Reset
    payload = {"task": "hard", "seed": 42}
    try:
        response = requests.post(f"{API_BASE_URL}/reset", json=payload)
        if response.status_code != 200:
            print(f"Error resetting: {response.text}")
            return
    except Exception as e:
        print(f"Connection Error: {e}")
        return
    
    step_result = response.json()
    obs = step_result["observation"]
    done = step_result["terminated"] or step_result.get("truncated", False)
    
    step_count = 0
    while not done:
        # Get action from LLM
        action = get_action_from_llm(obs)
        
        # Step
        response = requests.post(f"{API_BASE_URL}/step", json={"action": action})
        if response.status_code != 200:
            print(f"Error stepping: {response.text}")
            break
            
        step_result = response.json()
        obs = step_result["observation"]
        reward = step_result["reward"]
        done = step_result["terminated"] or step_result.get("truncated", False)
        
        print(f"[STEP] count={step_count} action={action} reward={reward:.4f}")
        step_count += 1
        
        if step_count > 150: # Safety break
            break
            
    print("[END]")

if __name__ == "__main__":
    run_episode()
