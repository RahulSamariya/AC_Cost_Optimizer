import sys
import os
import concurrent.futures

# Ensure the project directory is in the path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_dir)

from env import SupportTicketEnv
from tasks import get_easy_task, get_medium_task, get_hard_task, KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # Priority escalation
    if ticket.priority == "Urgent" or ticket.priority == "High":
        return Action(action_type=ActionType.ESCALATE, reasoning="Escalating high priority.")
    
    # Technical resolution
    if ticket.category == TicketCategory.TECHNICAL:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_001", reasoning="Resolving tech.")
    
    # Billing resolution
    if ticket.category == TicketCategory.BILLING:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_002", reasoning="Resolving billing.")
        
    # Default to escalation for Irrelevant to clear queue
    if ticket.category == TicketCategory.IRRELEVANT:
        return Action(action_type=ActionType.ESCALATE, reasoning="Escalating spam.")

    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Classifying.")

def run_task(name, get_task_func):
    tickets = get_task_func()
    env = SupportTicketEnv(tickets, KB_DATA)
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = heuristic_agent(obs)
        if not action: break
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
    score = grade_episode(env.state, total_reward)
    return {
        "name": name,
        "score": score,
        "reward": round(total_reward, 2),
        "tickets": len(tickets)
    }

if __name__ == "__main__":
    tasks = [
        ("Easy", get_easy_task),
        ("Medium", get_medium_task),
        ("Hard", get_hard_task)
    ]
    
    print(f"Launching {len(tasks)} tasks simultaneously...\n")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_task, name, func) for name, func in tasks]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            results.append(res)
            print(f"  [DONE] {res['name']} Task (Score: {res['score']})")

    print("\n" + "="*50)
    print(f"{'TASK NAME':<10} | {'SCORE':<7} | {'REWARD':<8} | {'TICKETS':<8}")
    print("-" * 50)
    for res in sorted(results, key=lambda x: x['name']):
        print(f"{res['name']:<10} | {res['score']:<7} | {res['reward']:<8} | {res['tickets']:<8}")
    print("="*50)
