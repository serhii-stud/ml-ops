import requests
import time
import random
import csv
import sys
from datetime import datetime

# --- CONFIGURATION ---
API_URL = "http://0.0.0.0:8000/predict"
TOTAL_REQUESTS = 30  # How many requests to generate
current_date = datetime.now().strftime("%Y-%m-%d")
OUTPUT_FILE = f"requests_history_{current_date}.csv"
MIN_DELAY = 0.1
MAX_DELAY = 0.5

# Dataset mapping: True Category -> List of phrases
# We need this structure to know the "Ground Truth" for future corrections
# Dataset mapping: True Category -> List of phrases

DATASET = {
    "card_arrival": [
        # Logistics / Status
        "Any update on when the card is supposed to arrive?",
        "My plastic hasn't shown up yet, what's going on?",
        "Still no envelope from you guys",
        "Has the shipment left the warehouse?",
        "When should I expect the card delivery?",

        # Anxiety / Informal
        "Been weeks… where is it?",
        "Mailbox empty again 😐",
        "Is this normal delivery time?",
        "Feels like the card is lost already",

        # Address / Tracking
        "I recently moved, could that affect delivery?",
        "Tracking link doesn't open",
        "Courier says completed but nothing received"
    ],

    "lost_or_stolen_card": [
        # Loss scenarios
        "Pretty sure my wallet is gone",
        "Someone must have taken my card",
        "Card disappeared overnight",
        "Can't find it anywhere",

        # Security-driven wording
        "I suspect card compromise",
        "Possible theft, need to act fast",
        "Someone tried to pay with my card",
        "Strange transactions showing up",

        # Urgent actions
        "Freeze everything now",
        "Block card immediately please",
        "I want to order a new one"
    ],

    "balance_check": [
        # Casual / Modern
        "How much cash do I even have?",
        "What’s left after yesterday?",
        "Is my account empty?",
        "Can I survive till payday?",

        # App / UX language
        "Show me my current funds",
        "Quick balance check",
        "Snapshot of my account",

        # Temporal
        "Balance after rent payment?",
        "Did the transfer increase my balance?",
        "Status after last transaction"
    ],

    "declined_card_payment": [
        # Physical world
        "Terminal rejected my card",
        "POS wouldn’t accept payment",
        "Cashier said transaction failed",
        "Payment bounced",

        # Digital / Subscription
        "Streaming service says payment issue",
        "Online checkout keeps failing",
        "My card isn’t working on apps",

        # Confusion / Debugging
        "There’s money but it still declines",
        "Why am I blocked from paying?",
        "Is my card restricted?"
    ],

    "country_support": [
        # Travel intent
        "Flying abroad, do I need to enable something?",
        "Can I swipe my card outside Europe?",
        "Using card while travelling",

        # Countries / Regions
        "Payments in Asia supported?",
        "Does this card work in Mexico?",
        "Non-EU card usage",

        # Fees / Limits
        "Extra charges overseas?",
        "Is FX applied automatically?",
        "Do I need travel mode?"
    ],

    "cancel_transfer": [
        # Immediate regret
        "Oops, sent money too fast",
        "I made a mistake with the transfer",
        "That payment shouldn’t have gone out",

        # Technical wording
        "Cancel pending transaction",
        "Reverse last transfer",
        "Is rollback possible?",

        # Fraud concern
        "I think I was tricked into sending money",
        "Transfer sent under false pretenses"
    ],

    "terminate_account": [
        # Neutral / Formal
        "Requesting account closure",
        "I would like to terminate my account",
        "Please proceed with closing my profile",

        # Emotional
        "This service no longer works for me",
        "I’m done banking here",
        "Time to move on",

        # Data & compliance
        "Delete all my personal records",
        "Confirm full account shutdown",
        "End customer relationship"
    ],

    "edit_personal_details": [
        # Identity
        "My legal name has changed",
        "Surname update required",
        "Incorrect personal info on file",

        # Contact data
        "New phone number, old one inactive",
        "Email change request",
        "Wrong address saved",

        # Compliance
        "Need to update identity details",
        "Adjust customer profile"
    ],

    "exchange_rate": [
        # Conversational
        "What’s today’s FX rate?",
        "How bad is the conversion?",
        "Am I getting ripped off on exchange?",

        # Comparative
        "Better than street exchange?",
        "Compare with bank rates",
        "Is this market rate?",

        # Specific intent
        "Convert EUR to JPY",
        "USD exchange pricing",
        "Foreign currency costs"
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