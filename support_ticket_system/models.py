from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class TicketCategory(str, Enum):
    TECHNICAL = "Technical"
    BILLING = "Billing"
    FEATURE_REQUEST = "Feature Request"
    IRRELEVANT = "Irrelevant"

class Priority(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    URGENT = "Urgent"

class TicketStatus(str, Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    RESOLVED = "Resolved"
    ESCALATED = "Escalated"

class ActionType(str, Enum):
    CLASSIFY = "classify"
    RESOLVE_WITH_KB = "resolve_with_kb_article"
    ESCALATE = "escalate"

class Ticket(BaseModel):
    id: str = Field(..., description="Unique ticket identifier")
    description: str = Field(..., description="Raw text of the customer issue")
    category: Optional[TicketCategory] = Field(None, description="Assigned category")
    priority: Priority = Field(Priority.MEDIUM, description="Ticket urgency level")
    sentiment: float = Field(0.5, description="Sentiment score from 0.0 (angry) to 1.0 (happy)")
    required_kb_id: Optional[str] = Field(None, description="The specific KB ID needed for resolution")
    status: TicketStatus = Field(TicketStatus.PENDING)

class Observation(BaseModel):
    active_ticket: Optional[Ticket] = Field(None, description="Full details of the current ticket")
    queue_backlog: int = Field(..., description="Remaining tickets in queue")
    agent_fatigue: float = Field(..., description="Current fatigue (0.0 to 1.0)")
    kb_preview: List[str] = Field(default_factory=list, description="Available KB IDs")

class Action(BaseModel):
    action_type: ActionType = Field(..., description="The type of action to perform")
    category: Optional[TicketCategory] = Field(None, description="Required for 'classify'")
    kb_id: Optional[str] = Field(None, description="Required for 'resolve_with_kb_article'")
    reasoning: str = Field(..., description="Internal thought process")

class State(BaseModel):
    all_tickets: List[Ticket]
    current_index: int = 0
    steps_taken: int = 0
    max_steps: int = 100
    kb_data: Dict[str, str]
