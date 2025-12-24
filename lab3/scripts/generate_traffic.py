import requests
import time
import random
import csv
import sys
from datetime import datetime

# --- CONFIGURATION ---
API_URL = "http://0.0.0.0:8000/predict"
TOTAL_REQUESTS = 20  # How many requests to generate
current_date = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"requests_history_{current_date}.csv"
MIN_DELAY = 0.1
MAX_DELAY = 0.5

# Dataset mapping: True Category -> List of phrases
# We need this structure to know the "Ground Truth" for future corrections
# Dataset mapping: True Category -> List of phrases
DATASET = {
    "card_arrival": [
        # Direct questions
        "Where is my card?",
        "When will the credit card arrive?",
        "Has my card been shipped yet?",
        # Status checks
        "Track my card delivery status",
        "Check delivery status",
        "Show me the tracking number for my card",
        # Time complaints
        "I ordered a card last week and it's still not here",
        "It's been 10 days, where is my visa?",
        "Waiting too long for the card delivery",
        "Is my card lost in the mail?"
    ],
    "lost_or_stolen_card": [
        # Emergency
        "Help, my card was stolen!",
        "Someone stole my wallet",
        "Emergency! Card theft",
        "I think I was pickpocketed",
        # Lost
        "I lost my wallet with my card",
        "I cannot find my visa card",
        "I dropped my card somewhere in the park",
        "My card is missing",
        # Action requests
        "Block my card immediately please",
        "Freeze my account, lost card",
        "Deactivate my card right now",
        "Need to cancel lost card"
    ],
    "balance_check": [
        # Simple
        "What is my current balance?",
        "How much money do I have left?",
        "Show me my account total",
        "Balance inquiry",
        # Specific
        "How much cash is available?",
        "Do I have enough for a $500 purchase?",
        "Check funds",
        "What's the remaining limit?",
        # App/UI references
        "I can't see my balance in the app",
        "Why is my balance not updating?",
        "Tell me my net worth"
    ],
    "declined_card_payment": [
        # Confusion
        "My payment was declined at the shop",
        "Why can't I buy this coffee?",
        "I have money, why did the transaction fail?",
        # Technical
        "Transaction rejected",
        "Card not working",
        "Payment error 404",
        "The terminal said declined",
        # Specific scenarios
        "Netflix payment didn't go through",
        "Got embarrassed at the restaurant, card rejected",
        "Online purchase failed",
        "Unable to pay for my ticket"
    ],
    "country_support": [
        # Travel plans
        "Do you support US payments?",
        "Can I use this in France?",
        "I am traveling to Japan next week",
        "Will my card work in Germany?",
        # Currency/Region
        "Do you support payments in USD?",
        "Is this card valid in the EU?",
        "Can I withdraw cash in Bali?",
        "Restricted countries list",
        "Using card abroad"
    ],
"cancel_transfer": [
        "I sent money to the wrong person, cancel it!",
        "Stop the transfer immediately",
        "I made a mistake in the transaction",
        "Can I reverse a payment?",
        "Please undo the last transfer",
        "I want to claim my money back",
        "Accidental transfer, help"
    ],
    "terminate_account": [
        "I want to close my account",
        "Terminate my contract",
        "I am leaving this bank",
        "How do I delete my profile?",
        "Close account and withdraw all funds",
        "I'm not happy with your service, goodbye",
        "Cancel my subscription and account"
    ],
    "edit_personal_details": [
        "I moved to a new house, how to change address?",
        "Update my phone number",
        "My last name changed after marriage",
        "I need to update my personal info",
        "Change billing address",
        "Incorrect date of birth in my profile",
        "Edit contact details"
    ],
    "exchange_rate": [
        "What is the current exchange rate for Euro?",
        "How much is 1 USD in GBP?",
        "Check currency rates",
        "Exchange rate for today",
        "Do you have good rates for Yen?",
        "Convert 100 dollars to euros",
        "Forex rates please"
    ]
}


def generate_traffic():
    print(f"🚀 Starting traffic generation to {API_URL}...")
    print(f"🎯 Goal: {TOTAL_REQUESTS} requests")
    print(f"💾 Saving results to: {OUTPUT_FILE}")
    print("-" * 40)

    success_count = 0
    fail_count = 0

    # Open CSV file for writing
    with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write Header
        writer.writerow(['request_id', 'timestamp', 'text', 'predicted_category', 'true_category', 'is_correct'])

        for i in range(1, TOTAL_REQUESTS + 1):
            # 1. Pick a random category (True Label) and a random text from it
            true_category = random.choice(list(DATASET.keys()))
            text = random.choice(DATASET[true_category])

            # 2. Prepare payload
            payload = {"text": text}

            try:
                start_time = time.time()
                # 3. Send Request
                response = requests.post(API_URL, json=payload)
                elapsed = time.time() - start_time

                # 4. Process Response
                if response.status_code == 200:
                    data = response.json()

                    req_id = data.get("request_id", "NO_ID")
                    predicted = data.get("category", "UNKNOWN")
                    timestamp = datetime.now().isoformat()

                    # Check if model was correct (for local stats)
                    is_correct = (predicted == true_category)
                    status_icon = "✅" if is_correct else "❌"

                    # 5. Save to CSV
                    writer.writerow([req_id, timestamp, text, predicted, true_category, is_correct])

                    print(f"[{i}/{TOTAL_REQUESTS}] {status_icon} ({elapsed:.2f}s) ID: {req_id} | Pred: {predicted}")
                    success_count += 1
                else:
                    print(f"[{i}/{TOTAL_REQUESTS}] ⚠️ Error {response.status_code}: {response.text}")
                    fail_count += 1

            except requests.exceptions.ConnectionError:
                print(f"[{i}/{TOTAL_REQUESTS}] 🚨 Connection Refused! Is the service running?")
                fail_count += 1
            except Exception as e:
                print(f"[{i}/{TOTAL_REQUESTS}] 💥 Unexpected error: {e}")
                fail_count += 1

            # 6. Sleep to simulate real user traffic
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))

    print("-" * 40)
    print(f"🏁 Done! Success: {success_count}, Failed: {fail_count}")
    print(f"📄 Check {OUTPUT_FILE} for details.")


if __name__ == "__main__":
    generate_traffic()