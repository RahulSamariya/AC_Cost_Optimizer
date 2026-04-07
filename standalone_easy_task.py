import random
import uuid
import copy
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# --- MODELS ---
class TicketPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TicketStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    ESCALATED = "ESCALATED"

class ActionType(str, Enum):
    CLASSIFY = "CLASSIFY"
    RESOLVE_WITH_KB = "RESOLVE_WITH_KB"
    ESCALATE = "ESCALATE"

class Ticket(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    id: str
    title: str
    description: str
    priority: TicketPriority
    category: str
    has_kb_solution: bool
    status: TicketStatus = TicketStatus.PENDING

class Action(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    action_type: ActionType
    ticket_id: str
    classified_priority: Optional[str] = None

class Observation(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    ticket_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    current_status: Optional[TicketStatus] = None
    step: int
    done: bool
    reward: float
    info: Dict[str, Any] = Field(default_factory=dict)

# --- REWARD LOGIC ---
def calculate_reward(action: Action, ticket: Ticket) -> float:
    reward = -0.5 # Penalty
    if action.action_type == ActionType.CLASSIFY:
        reward += 0.0
    elif action.action_type == ActionType.RESOLVE_WITH_KB:
        reward += 5.0 if ticket.has_kb_solution else -1.5
    elif action.action_type == ActionType.ESCALATE:
        reward += 1.0 if ticket.priority == TicketPriority.URGENT else -2.0 if ticket.priority == TicketPriority.LOW else 0.0
    return float(reward)

# --- ENVIRONMENT ---
class TicketEnv:
    def __init__(self, tickets: List[Ticket], episode_id: str):
        self.tickets = copy.deepcopy(tickets)
        self.episode_id = episode_id
        self.current_index = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False

    def reset(self) -> Observation:
        self.current_index = 0
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        return self._get_observation(reward=0.0)

    def step(self, action: Action) -> Observation:
        if self.done: return self._get_observation(reward=0.0)
        ticket = self.tickets[self.current_index]
        reward = calculate_reward(action, ticket)
        self.total_reward += reward
        self.step_count += 1
        if action.action_type == ActionType.CLASSIFY:
            ticket.status = TicketStatus.IN_PROGRESS
        elif action.action_type == ActionType.RESOLVE_WITH_KB:
            ticket.status = TicketStatus.RESOLVED
            self.current_index += 1
        elif action.action_type == ActionType.ESCALATE:
            ticket.status = TicketStatus.ESCALATED
            self.current_index += 1
        if self.current_index >= len(self.tickets): self.done = True
        return self._get_observation(reward=reward)

    def _get_observation(self, reward: float) -> Observation:
        if self.done: return Observation(step=self.step_count, done=True, reward=reward, info={"total_reward": self.total_reward})
        t = self.tickets[self.current_index]
        return Observation(ticket_id=t.id, title=t.title, description=t.description, category=t.category, current_status=t.status, step=self.step_count, done=False, reward=reward, info={"total_reward": self.total_reward})

# --- RUNNER ---
def run_standalone_easy_task():
    print("--- Running Standalone Easy Task ---")
    episode_id = str(uuid.uuid4())
    easy_tickets = [
        Ticket(id="E-001", title="Billing Login", description="Issue with billing.", priority=TicketPriority.LOW, category="billing", has_kb_solution=True),
        Ticket(id="E-002", title="Refund Request", description="Billing refund.", priority=TicketPriority.LOW, category="billing", has_kb_solution=True),
        Ticket(id="E-003", title="Update Card", description="Billing credit card.", priority=TicketPriority.LOW, category="billing", has_kb_solution=True),
    ]
    
    env = TicketEnv(easy_tickets, episode_id)
    obs = env.reset()
    
    while not obs.done:
        t_id = obs.ticket_id
        # Strategy: CLASSIFY then RESOLVE
        action_type = ActionType.CLASSIFY if obs.current_status == TicketStatus.PENDING else ActionType.RESOLVE_WITH_KB
        action = Action(action_type=action_type, ticket_id=t_id)
        obs = env.step(action)
        print(f"  Step {obs.step} | Ticket {t_id} | Action {action_type} | Reward {obs.reward:+.1f} | Total {obs.info['total_reward']:+.1f}")

    print(f"\n[COMPLETED] Total Reward: {obs.info['total_reward']}")

if __name__ == "__main__":
    run_standalone_easy_task()
