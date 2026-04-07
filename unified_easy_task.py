import uuid
from core.env_logic import TicketEnv
from tasks.generator import generate_tickets
from models.action import Action, ActionType
from graders.grader import grade_episode

def run_unified_easy_task():
    print("--- Running Unified Easy Task (In-Process) ---")
    
    # 1. Generate Easy Tickets
    tickets = generate_tickets(5, difficulty="easy")
    episode_id = str(uuid.uuid4())
    
    # 2. Initialize Environment directly
    env = TicketEnv(tickets, episode_id)
    obs = env.reset()
    
    print(f"  [START] Episode ID: {episode_id}")
    
    while not obs["done"]:
        ticket_id = obs["ticket_id"]
        status = obs["current_status"]
        
        # Strategy: Classify -> Resolve
        if status == "PENDING":
            action_type = ActionType.CLASSIFY
        else:
            action_type = ActionType.RESOLVE_WITH_KB

        action = Action(
            action_type=action_type,
            ticket_id=ticket_id
        )

        obs = env.step(action)
        print(f"    Step: {obs['step']} | Ticket: {ticket_id} | Action: {action_type} | Reward: {obs['reward']:.1f}")

    # 3. Final Report
    report = grade_episode(env.trajectory)
    print(f"\n  [COMPLETED] Performance Report:")
    print(f"    Total Reward: {report['total_reward']}")
    print(f"    Correct Resolves: {report['correct_resolves']}")
    print(f"    Accuracy: {report['accuracy']*100}%")

if __name__ == "__main__":
    run_unified_easy_task()
