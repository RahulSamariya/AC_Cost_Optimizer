import requests
import uuid
import time
import sys

API_URL = "http://127.0.0.1:7860"

def run_easy_task():
    print("--- Executing Easy Task (5 Billing Tickets) ---")
    episode_id = str(uuid.uuid4())
    
    try:
        # Reset with Easy Difficulty
        resp = requests.post(f"{API_URL}/reset", json={"episode_id": episode_id, "difficulty": "easy"})
        resp.raise_for_status()
        obs = resp.json()
        
        print(f"  [START] Episode ID: {episode_id}")
        
        while not obs["done"]:
            # Rule-based Agent Strategy:
            # 1. Classify first
            # 2. Since all Easy tickets have KB solutions, resolve immediately after
            ticket_id = obs["ticket_id"]
            status = obs["current_status"]
            
            if status == "PENDING":
                action_type = "CLASSIFY"
            else:
                action_type = "RESOLVE_WITH_KB"

            payload = {
                "episode_id": episode_id,
                "action": {
                    "action_type": action_type,
                    "ticket_id": ticket_id
                }
            }

            resp = requests.post(f"{API_URL}/step", json=payload)
            resp.raise_for_status()
            obs = resp.json()
            
            print(f"    Step: {obs['step']} | Ticket: {ticket_id} | Action: {action_type} | Reward: {obs['reward']:.1f}")

        # Fetch Final Trajectory and Grade
        report_resp = requests.get(f"{API_URL}/trajectory", params={"episode_id": episode_id})
        report = report_resp.json()["report"]
        print(f"\n  [COMPLETED] Easy Task Performance:")
        print(f"    Final Reward: {report['total_reward']}")
        print(f"    Correct Resolves: {report['correct_resolves']}")
        print(f"    Accuracy: {report['accuracy']*100}%")

    except requests.exceptions.ConnectionError:
        print(f"  [ERROR] Server not available at {API_URL}")
        sys.exit(1)

if __name__ == "__main__":
    time.sleep(2) # Give server time to boot
    run_easy_task()
