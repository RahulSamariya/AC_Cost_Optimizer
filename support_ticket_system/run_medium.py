import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SupportTicketEnv
from tasks import get_medium_task, KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # Priority escalation for Medium Task
    if ticket.priority == "High":
        return Action(action_type=ActionType.ESCALATE, reasoning="High priority escalation.")
    
    # Technical resolution
    if ticket.category == TicketCategory.TECHNICAL:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_001", reasoning="Resolving tech ticket.")
        
    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Classifying.")

def run_medium():
    print("--- Starting Medium Task Execution ---")
    tickets = get_medium_task()
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
        print(f"  Ticket: {current_id} | Priority: {obs.active_ticket.priority if obs.active_ticket else 'Done'} | Action: {action.action_type} | Reward: {reward}")
    
    score = grade_episode(env.state, total_reward)
    print(f"\nMedium Task Summary:")
    print(f"  Final Score: {score}")
    print(f"  Total Reward: {round(total_reward, 2)}")
    print(f"  Tickets Processed: {len(env.state.all_tickets)}")

if __name__ == "__main__":
    run_medium()
