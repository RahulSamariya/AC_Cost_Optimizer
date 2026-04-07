import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SupportTicketEnv
from factory import TaskFactory
from tasks import KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # Simple logic for random tasks
    if ticket.priority == "Urgent" or ticket.category == TicketCategory.IRRELEVANT:
        return Action(action_type=ActionType.ESCALATE, reasoning="Escalating high-prio or noise.")
    
    kb_map = {
        TicketCategory.TECHNICAL: "KB_001",
        TicketCategory.BILLING: "KB_002",
        TicketCategory.FEATURE_REQUEST: "KB_003"
    }
    
    if ticket.category in kb_map:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id=kb_map[ticket.category], reasoning="Resolving.")
    
    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Classifying.")

def run_random_session(num_tickets: int = 10):
    print(f"--- Generating {num_tickets} Random Tickets ---")
    tickets = TaskFactory.create_random_batch(num_tickets)
    
    env = SupportTicketEnv(tickets, KB_DATA)
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = heuristic_agent(obs)
        if not action: break
        
        ticket_id = obs.active_ticket.id
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Processed {ticket_id} -> Result: {reward:+.1f}")
        
    score = grade_episode(env.state, total_reward)
    print(f"\nRandom Simulation Summary:")
    print(f"  Tickets Cleared: {num_tickets}")
    print(f"  Final Score: {score}")
    print(f"  Total Reward: {round(total_reward, 2)}")

if __name__ == "__main__":
    run_random_session(10) # Change this number to run any amount of "real-world" tickets
