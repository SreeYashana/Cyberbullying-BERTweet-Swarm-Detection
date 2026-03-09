import json
import random
from datetime import datetime, timedelta

def random_metrics():
    return {
        "retweet_count": random.randint(0, 20),
        "reply_count": random.randint(0, 5),
        "like_count": random.randint(0, 15),
        "quote_count": random.randint(0, 3),
        "bookmark_count": random.randint(0, 3),
        "impression_count": random.randint(10, 500)
    }

insults = [
    "You're clueless.",
    "Nobody respects you.",
    "Stop embarrassing yourself.",
    "You're a complete failure.",
    "How are you this ignorant?",
    "Go away already.",
    "No one asked for your opinion.",
    "You're just spreading nonsense.",
    "It's painful watching you talk.",
    "You're making a fool of yourself."
]

accounts = []

for i in range(1, 101):
    tweet_time = datetime.utcnow() - timedelta(minutes=i)

    account = {
        "author_id": str(1000000000 + i),
        "username": f"user_{i}",
        "name": f"Test User {i}",
        "verified": False,
        "followers": random.randint(10, 5000),
        "bad_tweets": [
            {
                "tweet_id": str(5000000000 + i),
                "text": random.choice(insults),
                "created_at": tweet_time.isoformat() + "Z",
                "metrics": random_metrics(),
                "model": {
                    "label": "CYBERBULLYING",
                    "confidence": round(random.uniform(0.90, 0.99), 4),
                    "flagged": True,
                    "reasons": ["Synthetic abusive example"]
                }
            }
        ]
    }

    accounts.append(account)

data = {
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "mode": "synthetic_test_data",
    "tweets_collected": 100,
    "bad_tweets_kept": 100,
    "accounts": accounts
}

with open("accounts_100.json", "w") as f:
    json.dump(data, f, indent=2)

print("✅ accounts_100.json generated successfully.")
