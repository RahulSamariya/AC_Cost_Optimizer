import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SupportTicketEnv
from tasks import get_easy_task, get_medium_task, get_hard_task, KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # Escalate Urgent OR Irrelevant (to advance queue)
    if ticket.priority == "Urgent" or ticket.category == TicketCategory.IRRELEVANT:
        return Action(action_type=ActionType.ESCALATE, reasoning="Terminal action for high priority or spam.")
    
    if ticket.category == TicketCategory.TECHNICAL:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_001", reasoning="Resolving via KB.")
    
    if ticket.category == TicketCategory.BILLING:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_002", reasoning="Resolving via KB.")
        
    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Classifying.")

def run_task(name, tickets):
    print(f"\n--- Running {name} Task ---")
    env = SupportTicketEnv(tickets, KB_DATA)
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = heuristic_agent(obs)
        if not action: break
        
        current_id = obs.active_ticket.id
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Processed: {current_id} | Action: {action.action_type} | Reward: {reward}")
    
    score = grade_episode(env.state, total_reward)
    print(f"Final Score for {name}: {score} | Total Reward: {round(total_reward, 2)}")

if __name__ == "__main__":
    run_task("Easy", get_easy_task())
    run_task("Medium", get_medium_task())
    run_task("Hard", get_hard_task())
