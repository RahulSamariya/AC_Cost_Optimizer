import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SupportTicketEnv
from tasks import get_easy_task, KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # Billing resolution for Easy Task
    if ticket.category == TicketCategory.BILLING:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_002", reasoning="Resolving billing ticket.")
        
    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Classifying.")

def run_easy():
    print("--- Starting Easy Task Execution ---")
    tickets = get_easy_task()
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
        print(f"  Ticket: {current_id} | Action: {action.action_type} | Reward: {reward}")
    
    score = grade_episode(env.state, total_reward)
    print(f"\nEasy Task Summary:")
    print(f"  Final Score: {score}")
    print(f"  Total Reward: {round(total_reward, 2)}")
    print(f"  Tickets Resolved: {len(env.state.all_tickets)}")

if __name__ == "__main__":
    run_easy()
