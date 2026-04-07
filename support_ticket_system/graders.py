def grade_episode(state, total_reward) -> float:
    total = len(state.all_tickets)
    if total == 0: return 0.0
    resolved = len([t for t in state.all_tickets if t.status == "Resolved"])
    escalated = len([t for t in state.all_tickets if t.status == "Escalated"])
    
    success_rate = (resolved + escalated) / total
    efficiency = max(0, 1.0 - (state.steps_taken / (total * 2)))
    
    score = (success_rate * 0.7) + (efficiency * 0.3)
    return round(score, 2)
