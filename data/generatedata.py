import random

# Define the vocabulary
actions = ["PURCHASE", "PAYMENT AT", "POS", "REFUND", "TRANSFER TO"]
merchants = ["Amazon", "Starbucks", "IKEA", "H&M", "McDonalds", "Apple Store", "Subway", "REWE", "Lidl", "Zara", "Netflix", "Aldi", "COOP", "Migros"]
locations = ["Zurich", "Basel", "Bülach", "Oerlikon", "Winkel", "Bern", "Geneve", "Oftringen", "Berlin", "Paris", "London", "New York", "Tokyo", "Munich", "Rome"]

lines = []

for _ in range(10000):  # generate 10,000 lines
    action = random.choice(actions)
    merchant = random.choice(merchants)
    
    # Optionally append a location
    if random.random() < 0.3:  # 30% chance
        merchant += f" {random.choice(locations)}"
    
    # Optionally append a dollar/euro amount
    if random.random() < 0.3:  # 30% chance
        amount = random.randint(1, 500)
        merchant += f" ${amount}"
    
    lines.append(f"{action} {merchant}")

# Save to file
with open("synthetic_transactions.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Synthetic dataset created: synthetic_transactions.txt (10k lines)")
