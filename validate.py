import requests
import sys

BASE_URL = "http://localhost:8000"

def validate_server():
    print("--- Validating OpenEnv Server ---")
    
    # 1. Reset check
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"seed": 42})
    except requests.exceptions.ConnectionError:
        print("FAIL: Server not reachable on http://localhost:8000")
        return False
        
    if r.status_code != 200:
        print(f"FAIL: /reset returned {r.status_code}")
        return False
    
    obs = r.json().get("observation")
    if not isinstance(obs, list) or len(obs) != 6:
        print(f"FAIL: Invalid observation schema: {obs}")
        return False
    print("✅ /reset schema is correct.")

    # 2. Step check
    r = requests.post(f"{BASE_URL}/step", json={"action": 2})
    if r.status_code != 200:
        print(f"FAIL: /step returned {r.status_code}")
        return False
        
    res = r.json()
    required_keys = ["observation", "reward", "done", "info"]
    for key in required_keys:
        if key not in res:
            print(f"FAIL: /step response missing key '{key}'")
            return False
    
    print("✅ /step schema is correct.")
    
    # 3. Full Episode Length Check
    print("--- Running Full Episode Validation ---")
    done = False
    steps = 0
    while not done and steps < 200:
        r = requests.post(f"{BASE_URL}/step", json={"action": 0})
        res = r.json()
        done = res["done"]
        steps += 1
    
    if steps != 143: # We already took 1 step above, so total 144
        print(f"FAIL: Episode ended at step {steps+1}, expected 144.")
        return False
        
    print(f"✅ Episode length is correct (144 steps).")
    print("--- VALIDATION SUCCESSFUL ---")
    return True

if __name__ == "__main__":
    if not validate_server():
        sys.exit(1)
