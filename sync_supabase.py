"""Sync news digest to Supabase."""
import json, os, sys
from supabase import create_client

env = {}
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")

SUPABASE_URL = env.get("SUPABASE_URL", "https://pqwmmwvglxsvuzndkrgj.supabase.co")
SUPABASE_KEY = env.get("SUPABASE_KEY", "")
if not SUPABASE_KEY:
    print("SUPABASE_KEY not set"); sys.exit(1)

client = create_client(SUPABASE_URL, SUPABASE_KEY)

with open("output/daily_digest.json", "r", encoding="utf-8") as f:
    data = json.load(f)

items = data.get("items", [])
stats = data.get("stats", {})
date = data.get("date", "")
print(f"Syncing {len(items)} items for {date}...")

synced = 0
for item in items:
    row = {
        "id": item.get("id", ""),
        "title": item.get("title", "")[:500],
        "headline_fa": item.get("headline_fa", ""),
        "summary_fa": item.get("summary_fa", ""),
        "why_it_matters": item.get("why_it_matters", ""),
        "source": item.get("source", ""),
        "url": item.get("url", ""),
        "importance_score": item.get("importance_score", 0),
        "viral_potential": item.get("viral_potential", "low"),
        "categories": item.get("categories", []),
        "tags": item.get("tags", []),
        "opportunity": item.get("opportunity", "no"),
        "published_at": item.get("published_at", ""),
        "base_category": item.get("base_category", ""),
    }
    try:
        client.table("news_items").upsert(row).execute()
        synced += 1
    except Exception as e:
        print(f"  err {row['id']}: {e}")

print(f"Synced {synced}/{len(items)} items")

try:
    client.table("digest_stats").upsert({
        "date": date,
        "total_sources": stats.get("total_sources", 0),
        "total_fetched": stats.get("total_fetched", 0),
        "total_displayed": stats.get("total_displayed", 0),
        "top_score": stats.get("top_score", 0),
        "cost": 0.0094,
    }).execute()
    print("Stats synced")
except Exception as e:
    print(f"Stats err: {e}")

print("Done!")
