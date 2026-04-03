import numpy as np

def grade_episode(trajectory: list) -> dict:
    """
    Grades an episode with high sensitivity to energy and cost.
    Returns a dict with a 'total' key.
    All scores are clipped to [0.0, 1.0].
    """
    if not trajectory:
        return {"total": 0.0, "error": "Empty trajectory"}

    comfort_scores = [step['info'].get('comfort', 0.0) for step in trajectory]
    energies = [step['info'].get('energy', 0.0) for step in trajectory]
    
    avg_comfort = float(np.mean(comfort_scores))
    total_energy = float(np.sum(energies))
    
    # 1. Comfort Grade (0 to 1)
    comfort_grade = float(np.clip(avg_comfort, 0.0, 1.0))
    
    # 2. Energy Grade (Sensitive)
    # Using Option B: 1 / (1 + total_energy)
    # Scaled such that 10kWh results in a decent score.
    energy_grade = float(np.clip(10.0 / (10.0 + total_energy), 0.0, 1.0))
    
    # 3. Final Total Score
    # 70% Comfort, 30% Energy efficiency
    total_score = (0.7 * comfort_grade) + (0.3 * energy_grade)
    
    return {
        "total": float(total_score),
        "avg_comfort": avg_comfort,
        "total_energy_kwh": total_energy,
        "comfort_grade": comfort_grade,
        "energy_grade": energy_grade
    }
