from models import Ticket, TicketCategory, Priority

KB_DATA = {
    "KB_001": "Technical reset procedure.",
    "KB_002": "Billing cycle information.",
    "KB_003": "Feature request intake form."
}

def get_easy_task():
    return [
        Ticket(id="E1", description="Refund my money.", category=TicketCategory.BILLING, priority=Priority.LOW, required_kb_id="KB_002"),
        Ticket(id="E2", description="Change my plan.", category=TicketCategory.BILLING, priority=Priority.MEDIUM, required_kb_id="KB_002"),
    ]

def get_medium_task():
    return [
        Ticket(id=f"M{i}", description=f"Issue {i}", category=TicketCategory.TECHNICAL, priority=Priority.HIGH if i % 2 == 0 else Priority.MEDIUM, required_kb_id="KB_001")
        for i in range(5)
    ]

def get_hard_task():
    tickets = []
    for i in range(10):
        if i % 3 == 0:
            tickets.append(Ticket(id=f"H{i}", description="Spam", category=TicketCategory.IRRELEVANT, priority=Priority.LOW, required_kb_id=None))
        else:
            tickets.append(Ticket(id=f"H{i}", description="Complex Tech", category=TicketCategory.TECHNICAL, priority=Priority.URGENT, required_kb_id="KB_001"))
    return tickets
