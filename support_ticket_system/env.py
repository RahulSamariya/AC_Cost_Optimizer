from typing import Tuple, Dict, Any, List
from models import State, Observation, Action, TicketStatus, ActionType, Ticket

class SupportTicketEnv:
    def __init__(self, tickets: List[Ticket], kb_data: Dict[str, str], max_steps: int = 100):
        self.initial_tickets = tickets
        self.kb_data = kb_data
        self.max_steps = max_steps
        self.state = None

    def reset(self) -> Observation:
        self.state = State(
            all_tickets=[t.model_copy(deep=True) for t in self.initial_tickets],
            current_index=0,
            steps_taken=0,
            max_steps=self.max_steps,
            kb_data=self.kb_data
        )
        return self._get_obs()

    def _get_obs(self) -> Observation:
        active = None
        if self.state.current_index < len(self.state.all_tickets):
            active = self.state.all_tickets[self.state.current_index]
        
        return Observation(
            active_ticket=active,
            queue_backlog=len(self.state.all_tickets) - self.state.current_index,
            agent_fatigue=min(1.0, self.state.steps_taken / self.state.max_steps),
            kb_preview=list(self.state.kb_data.keys())
        )

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.state.current_index >= len(self.state.all_tickets):
            return self._get_obs(), 0.0, True, {"info": "Queue empty"}

        ticket = self.state.all_tickets[self.state.current_index]
        self.state.steps_taken += 1
        reward = -0.5  # Step penalty
        
        if action.action_type == ActionType.CLASSIFY:
            if action.category == ticket.category:
                reward += 1.0
            ticket.status = TicketStatus.IN_PROGRESS
            
        elif action.action_type == ActionType.RESOLVE_WITH_KB:
            if action.kb_id == ticket.required_kb_id:
                reward += 5.0
                ticket.status = TicketStatus.RESOLVED
                self.state.current_index += 1
            else:
                reward -= 1.5
                ticket.sentiment = max(0.0, ticket.sentiment - 0.1)

        elif action.action_type == ActionType.ESCALATE:
            if ticket.priority in ["High", "Urgent"]:
                reward += 1.0
            else:
                reward -= 2.0
            ticket.status = TicketStatus.ESCALATED
            self.state.current_index += 1

        done = (self.state.current_index >= len(self.state.all_tickets)) or (self.state.steps_taken >= self.state.max_steps)
        return self._get_obs(), reward, done, {}
