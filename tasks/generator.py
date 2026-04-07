import random
import uuid
from typing import List, Optional
from models.observation import Ticket, TicketPriority

CATEGORIES = ["auth", "billing", "infra", "product", "security"]

def generate_tickets(n: int = 10, difficulty: Optional[str] = None) -> List[Ticket]:
    tickets = []
    for i in range(n):
        if difficulty == "easy":
            # Easy: Always Billing, Low Priority, Always has KB solution
            priority = TicketPriority.LOW
            category = "billing"
            has_kb = True
        else:
            priority = random.choice(list(TicketPriority))
            category = random.choice(CATEGORIES)
            has_kb = random.choice([True, False])
        
        tickets.append(Ticket(
            id=f"T-{uuid.uuid4().hex[:6].upper()}",
            title=f"{category.capitalize()} Issue - {i}",
            description=f"Standard request for {category} assistance.",
            priority=priority,
            category=category,
            has_kb_solution=has_kb
        ))
    return tickets
