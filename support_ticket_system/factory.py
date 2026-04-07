import random
from models import Ticket, TicketCategory, Priority, TicketStatus

class TaskFactory:
    """
    Utility to generate random, diverse ticket data for infinite RL training
    or stress-testing the support environment.
    """
    
    @staticmethod
    def create_random_batch(count: int = 5) -> list[Ticket]:
        # Realistic sample descriptions for each category
        samples = {
            TicketCategory.TECHNICAL: [
                "My app crashes on startup.",
                "VPN connection failed with error 404.",
                "The dashboard is not loading data.",
                "API key is rejected even though it is valid."
            ],
            TicketCategory.BILLING: [
                "I was overcharged for my last subscription.",
                "How do I update my payment method?",
                "Refund request for order #12345.",
                "I need a PDF copy of my last invoice."
            ],
            TicketCategory.FEATURE_REQUEST: [
                "Please add a Dark Mode to the UI.",
                "I want to export my reports to Excel.",
                "Integration with Microsoft Teams would be great."
            ],
            TicketCategory.IRRELEVANT: [
                "WIN A FREE IPHONE NOW!",
                "This is a test message.",
                "Hello, I just wanted to say hi."
            ]
        }

        # Mapping categories to the correct KB IDs in our environment
        kb_map = {
            TicketCategory.TECHNICAL: "KB_001",
            TicketCategory.BILLING: "KB_002",
            TicketCategory.FEATURE_REQUEST: "KB_003",
            TicketCategory.IRRELEVANT: None
        }

        tickets = []
        for i in range(count):
            category = random.choice(list(TicketCategory))
            priority = random.choice(list(Priority))
            desc = random.choice(samples[category])
            
            tickets.append(Ticket(
                id=f"RND-{i:03d}",
                description=desc,
                category=category,
                priority=priority,
                sentiment=random.uniform(0.2, 0.9),
                required_kb_id=kb_map[category],
                status=TicketStatus.PENDING
            ))
        return tickets
