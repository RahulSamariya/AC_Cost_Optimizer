import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env import SupportTicketEnv
from tasks import get_hard_task, KB_DATA
from models import Action, ActionType, TicketCategory
from graders import grade_episode

def heuristic_agent(obs):
    ticket = obs.active_ticket
    if not ticket: return None
    
    # TERMINAL ACTIONS (Move to next ticket)
    # 1. Escalate Urgent priority (as required by task)
    if ticket.priority == "Urgent":
        return Action(action_type=ActionType.ESCALATE, reasoning="Urgent ticket - Escalating to human tier.")
    
    # 2. Escalate Irrelevant/Spam to clear the queue
    if ticket.category == TicketCategory.IRRELEVANT:
        return Action(action_type=ActionType.ESCALATE, reasoning="Spam detected - Escalating to clear queue.")
    
    # 3. Resolve Technical issues
    if ticket.category == TicketCategory.TECHNICAL:
        return Action(action_type=ActionType.RESOLVE_WITH_KB, kb_id="KB_001", reasoning="Resolving complex tech issue.")
        
    # NON-TERMINAL ACTION
    return Action(action_type=ActionType.CLASSIFY, category=ticket.category, reasoning="Initial classification.")

def run_hard():
    print("--- Starting Hard Task Execution ---")
    tickets = get_hard_task()
    env = SupportTicketEnv(tickets, KB_DATA)
    obs = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = heuristic_agent(obs)
        if not action: break
        
        current_id = obs.active_ticket.id
        current_priority = obs.active_ticket.priority
        current_category = obs.active_ticket.category
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        print(f"  Processed: {current_id} | Cat: {current_category.value:10} | Prio: {current_priority.value:7} | Action: {action.action_type:15} | Reward: {reward:5.1f}")
    
    score = grade_episode(env.state, total_reward)
    print(f"\nHard Task Summary:")
    print(f"  Final Score: {score}")
    print(f"  Total Reward: {round(total_reward, 2)}")
    print(f"  Tickets Processed: {len(env.state.all_tickets)}")

if __name__ == "__main__":
    run_hard()
