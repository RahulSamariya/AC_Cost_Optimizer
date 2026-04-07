import random

class PricingSchedule:
    """
    Time-of-Use electricity pricing schedule.
    """
    def __init__(self, mode='tou'):
        self.mode = mode
        self.jitter = 1.0
        
        # Base schedule: hours and rates
        # off-peak (0-7): 0.08, mid-peak (7-17): 0.15, on-peak (17-22): 0.35, off-peak (22-24): 0.10
        self.schedule = [
            (0, 7, 0.08),
            (7, 17, 0.15),
            (17, 22, 0.35),
            (22, 24, 0.10)
        ]

    def reset_with_jitter(self):
        # ±20% jitter on price each episode reset
        self.jitter = random.uniform(0.8, 1.2)

    def get_price(self, hour_float: float) -> float:
        hour = hour_float % 24
        
        if self.mode == 'flat':
            return 0.12 * self.jitter
        
        base_rate = 0.12
        for start, end, rate in self.schedule:
            if start <= hour < end:
                base_rate = rate
                break
        
        if self.mode == 'dynamic' and random.random() < 0.05:
            # 5% chance of price spike in dynamic mode
            base_rate *= 3.0
            
        return float(base_rate * self.jitter)

    def get_next_hour_price(self, hour_float: float) -> float:
        return self.get_price(hour_float + 1.0)

    def is_peak_soon(self, hour_float: float, window_hours=2) -> bool:
        """
        Check if a high-price period (>= 0.30) is coming within the window.
        """
        for i in range(1, int(window_hours * 2) + 1):
            if self.get_price(hour_float + (i * 0.5)) >= 0.30:
                return True
        return False
