import argparse
import json
import copy
from cool_budget_ai.tasks.task_configs import make_env
from cool_budget_ai.baselines.smart_heuristic import SmartHeuristicAgent
from cool_budget_ai.graders.grader import grade_episode

def main():
    parser = argparse.ArgumentParser(description="Evaluate a baseline agent on the Cool Budget AI environment.")
    parser.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"],
                        help="The task difficulty level (default: easy)")
    parser.add_argument("--render", action="store_true", help="Render the environment steps")
    args = parser.parse_args()

    print(f"--- Running evaluation on TASK: {args.task.upper()} ---")

    # 1. Initialize environment and agent
    env = make_env(args.task)
    agent = SmartHeuristicAgent()
    
    obs = env.reset()
    trajectory = []
    total_reward = 0
    
    # 2. Run episode
    done = False
    while not done:
        current_info = {
            "price": env._prices[env.current_step],
            "occupancy": env._occupancies[env.current_step]
        }
        action = agent.select_action(env.state_obj, current_info)
        
        res = env.step(action)
        total_reward += res.reward
        
        trajectory.append({
            "state": copy.deepcopy(env.state_obj),
            "action": action,
            "reward": res.reward,
            "info": res.info
        })
        
        if args.render:
            # Manually print as we don't have gym.render
            print(f"Step: {env.current_step} | CPU Temp: {env.state_obj.cpu_temp:.2f}°C | "
                  f"Comfort: {res.info['comfort']:.2f} | Energy: {res.info['energy']:.2f} kWh")
            
        done = res.terminated or res.truncated

    # 3. Grade the episode
    results = grade_episode(trajectory)
    
    print("\n--- EVALUATION RESULTS ---")
    print(json.dumps(results, indent=2))
    print(f"\nFinal Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
