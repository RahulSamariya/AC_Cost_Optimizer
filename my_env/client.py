import requests
from typing import Tuple, Dict, Any
from .models import Action, Observation

class TicketEnvClient:
    """
    Client for interacting with the Support Ticket Environment over HTTP.
    Matches the standard Gymnasium/OpenEnv interface for easy agent integration.
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def reset(self) -> Observation:
        """Call server to reset state and get initial observation."""
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return Observation(**response.json())

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Send action to server and return the new observation and reward."""
        # Convert Pydantic model to dict for JSON serialization
        action_data = action.model_dump()
        
        response = requests.post(f"{self.base_url}/step", json=action_data)
        response.raise_for_status()
        
        data = response.json()
        obs = Observation(**data["observation"])
        return obs, data["reward"], data["done"], data["info"]
