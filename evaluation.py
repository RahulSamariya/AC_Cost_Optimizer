import numpy as np
import json
import copy
from cool_budget_ai.tasks.task_configs import make_env
from cool_budget_ai.baselines.smart_heuristic import SmartHeuristicAgent
from cool_budget_ai.graders.grader import grade_episode

def run_evaluation(agent, task_name, episodes=3):
    """
    Runs evaluation for a specific task over multiple episodes.
    """
    env = make_env(task_name)
    all_results = []
    
    for i in range(episodes):
        obs = env.reset(seed=42 + i)
        trajectory = []
        done = False
        
        while not done:
            current_info = {
                "price": env._prices[env.current_step],
                "occupancy": env._occupancies[env.current_step]
            }
            
            action = agent.select_action(env.state, current_info)
            res = env.step(action)
            
            trajectory.append({
                "state": copy.deepcopy(env.state),
                "action": action,
                "reward": res.reward,
                "info": res.info
            })
            done = res.done
            
        res_grade = grade_episode(trajectory)
        res_grade["total_reward"] = float(np.sum([s['reward'] for s in trajectory]))
        all_results.append(res_grade)
        
    avg_res = {k: np.mean([r[k] for r in all_results]) for k in all_results[0].keys()}
    return avg_res

def main():
    agent = SmartHeuristicAgent()
    
    print("--- Running Evaluation: EASY vs HARD ---")
    easy_results = run_evaluation(agent, "easy")
    hard_results = run_evaluation(agent, "hard")
    
    print("\nEASY TASK (Averaged):")
    print(json.dumps(easy_results, indent=2))
    
    print("\nHARD TASK (Averaged):")
    print(json.dumps(hard_results, indent=2))
    
    cost_diff = hard_results['total_cost_usd'] - easy_results['total_cost_usd']
    energy_diff = hard_results['total_energy_kwh'] - easy_results['total_energy_kwh']
    score_diff = hard_results['total_score'] - easy_results['total_score']
    
    print(f"\nCost Difference (Hard-Easy):   ${cost_diff:.4f}")
    print(f"Energy Difference (Hard-Easy): {energy_diff:.4f} kWh")
    print(f"Score Difference (Hard-Easy):  {score_diff:.4f}")
    
    if cost_diff <= 0:
        print("⚠️  WARNING: Hard environment is NOT more expensive than Easy.")
    if easy_results['total_score'] - hard_results['total_score'] < 0.05:
        print("⚠️  WARNING: Environment not sufficiently challenging.")
    else:
        print("✅ SUCCESS: Hard environment provides a significant challenge.")

if __name__ == "__main__":
    main()
