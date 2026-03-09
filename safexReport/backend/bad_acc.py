# discover_bad_accounts_model.py
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import tweepy
import requests
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------
# CONFIG
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

OUT_JSON = os.path.join(DATA_DIR, "bad_accounts.json")

# keyword query starter (to fetch candidate tweets)
BAD_WORDS = ["fuck", "shit", "asshole", "idiot", "stupid", "ugly", "loser", "shut up", "go die"]

# Limits you asked
MAX_RESULTS_PER_PAGE = 10     # pull only 10 tweets from X
MAX_BAD_PROFILES = 10         # only 10 profiles
MAX_BAD_TWEETS = 10           # only 10 bad tweets total

# Model API (your FastAPI must be running)
MODEL_API = os.getenv("MODEL_API", "http://127.0.0.1:8000/predict")

# If your model returns a different label name, add here
BAD_LABEL_KEYWORDS = ["CYBERBULLYING", "TOXIC", "ABUSIVE", "HATE"]
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.80"))  # optional


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_query(bad_words: List[str]) -> str:
    # OR query: ("word1" OR "word2" OR ...)
    parts = [f'"{w}"' for w in bad_words]
    return " OR ".join(parts) + " -is:retweet lang:en"


def extract_reset_time(err: Exception) -> str:
    """
    Try to read x-rate-limit-reset header from Tweepy TooManyRequests.
    """
    try:
        resp = getattr(err, "response", None)
        if resp is not None:
            headers = resp.headers
            reset = headers.get("x-rate-limit-reset")
            if reset:
                reset_ts = int(reset)
                return datetime.fromtimestamp(reset_ts).strftime("%Y-%m-%d %H:%M:%S")
    except:
        pass
    return "unknown (check later)"


def call_model_api(text: str) -> Optional[Dict[str, Any]]:
    """
    Calls your FastAPI /predict endpoint:
      POST { "text": "..." }
    Expected response (example):
      { "label": "...", "confidence": 0.91, "flagged": true, "reasons": [...], ... }
    """
    try:
        r = requests.post(
            MODEL_API,
            headers={"Content-Type": "application/json"},
            json={"text": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("⚠️ Model API error:", str(e))
        return None


def is_bad_prediction(pred: Dict[str, Any]) -> bool:
    """
    Decide if tweet is bad using model output.
    """
    label = str(pred.get("label", "")).upper()
    flagged = bool(pred.get("flagged", False))
    conf = float(pred.get("confidence", 0.0) or 0.0)

    # If your API already sets flagged correctly, this is enough:
    if flagged and conf >= CONF_THRESHOLD:
        return True

    # fallback: label-based decision
    if any(k in label for k in BAD_LABEL_KEYWORDS) and conf >= CONF_THRESHOLD:
        return True

    return False


def main():
    bearer = (
        os.environ.get("X_BEARER_TOKEN")
        or os.environ.get("BEARER_TOKEN")
        or os.environ.get("TWITTER_BEARER_TOKEN")
        or os.environ.get("barrer_token")  # your env sometimes uses this
    )

    if not bearer:
        print("❌ Missing BEARER_TOKEN (or X_BEARER_TOKEN / barrer_token) in env.")
        return

    # quick check model api reachable
    print("🔌 Model API:", MODEL_API)

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=False)

    query = build_query(BAD_WORDS)
    print("🔎 Searching X globally for candidate tweets...")
    print("Query:", query)
    print(f"Limits => pull:{MAX_RESULTS_PER_PAGE} tweets | keep:{MAX_BAD_TWEETS} bad tweets | profiles:{MAX_BAD_PROFILES}")

    accounts: Dict[str, Dict[str, Any]] = {}
    tweets_collected = 0
    bad_tweets_kept = 0

    try:
        # We fetch only ONE page of 10 tweets (as you asked)
        resp = client.search_recent_tweets(
            query=query,
            max_results=MAX_RESULTS_PER_PAGE,
            tweet_fields=["author_id", "created_at", "lang", "public_metrics", "text"],
            expansions=["author_id"],
            user_fields=["username", "name", "public_metrics", "verified"],
        )

        if not resp.data:
            out = {
                "generated_at": datetime.utcnow().isoformat(),
                "mode": "global_search_recent_tweets + model_filter",
                "query": query,
                "max_results_per_page": MAX_RESULTS_PER_PAGE,
                "tweets_collected": 0,
                "bad_tweets_kept": 0,
                "accounts": [],
                "note": "No tweets returned for this query right now."
            }
            write_json(OUT_JSON, out)
            print(f"✅ Saved: {OUT_JSON}")
            return

        # Map users by id
        users_by_id = {}
        if resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                users_by_id[str(u.id)] = u

        # Loop tweets, run model, keep only bad
        for t in resp.data:
            if bad_tweets_kept >= MAX_BAD_TWEETS:
                break
            if len(accounts) >= MAX_BAD_PROFILES:
                break

            tweets_collected += 1
            text = (t.text or "").strip()
            if not text:
                continue

            pred = call_model_api(text)
            if not pred:
                continue

            if not is_bad_prediction(pred):
                continue  # skip non-bad tweets

            # ✅ Bad tweet accepted
            bad_tweets_kept += 1

            uid = str(t.author_id)
            u = users_by_id.get(uid)

            # extract followers count safely
            followers = 0
            if u and getattr(u, "public_metrics", None):
                try:
                    followers = u.public_metrics.get("followers_count", 0)
                except:
                    followers = 0

            if uid not in accounts:
                # Stop adding new profiles if already reached limit
                if len(accounts) >= MAX_BAD_PROFILES:
                    continue

                accounts[uid] = {
                    "author_id": uid,
                    "username": getattr(u, "username", None) if u else None,
                    "name": getattr(u, "name", None) if u else None,
                    "verified": getattr(u, "verified", False) if u else False,
                    "followers": followers,
                    "bad_tweets": []
                }

            accounts[uid]["bad_tweets"].append({
                "tweet_id": str(t.id),
                "text": text,
                "created_at": str(t.created_at) if t.created_at else None,
                "metrics": t.public_metrics or {},
                "model": {
                    "label": pred.get("label"),
                    "confidence": pred.get("confidence"),
                    "flagged": pred.get("flagged"),
                    "reasons": pred.get("reasons", [])
                }
            })

        # Build leaderboard
        leaderboard = sorted(accounts.values(), key=lambda x: len(x["bad_tweets"]), reverse=True)

        out = {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "global_search_recent_tweets + model_filter",
            "query": query,
            "max_results_per_page": MAX_RESULTS_PER_PAGE,
            "tweets_collected": tweets_collected,
            "bad_tweets_kept": bad_tweets_kept,
            "max_bad_profiles": MAX_BAD_PROFILES,
            "max_bad_tweets": MAX_BAD_TWEETS,
            "accounts": leaderboard
        }

        write_json(OUT_JSON, out)
        print(f"✅ Saved: {OUT_JSON}")
        print(f"Tweets collected: {tweets_collected} | Bad tweets kept: {bad_tweets_kept} | Bad profiles: {len(leaderboard)}")

    except tweepy.errors.TooManyRequests as e:
        reset_time = extract_reset_time(e)
        out = {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "global_search_recent_tweets + model_filter",
            "query": query,
            "tweets_collected": tweets_collected,
            "bad_tweets_kept": bad_tweets_kept,
            "accounts": sorted(accounts.values(), key=lambda x: len(x["bad_tweets"]), reverse=True),
            "error": "429 Too Many Requests",
            "try_again_after": reset_time
        }
        write_json(OUT_JSON, out)
        print("⚠️ Rate limit hit (429). Saved whatever was collected so far.")
        print(f"✅ Saved: {OUT_JSON}")
        print(f"Try again after: {reset_time}")

    except Exception as e:
        out = {
            "generated_at": datetime.utcnow().isoformat(),
            "mode": "global_search_recent_tweets + model_filter",
            "query": query,
            "tweets_collected": tweets_collected,
            "bad_tweets_kept": bad_tweets_kept,
            "accounts": sorted(accounts.values(), key=lambda x: len(x["bad_tweets"]), reverse=True),
            "error": str(e)
        }
        write_json(OUT_JSON, out)
        print("❌ Unexpected error. Saved debug output to JSON anyway.")
        print(f"✅ Saved: {OUT_JSON}")
        raise


if __name__ == "__main__":
    main()
