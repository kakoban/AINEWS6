"""
AI Daily Digest — Telegram Sender (HTML + RTL + date)
"""
import json, requests, time, sys, os
from datetime import datetime, timezone, timedelta

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
env = {}
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")

bot_token = env.get("TELEGRAM_BOT_TOKEN", "")
chat_id = env.get("TELEGRAM_CHAT_ID", "")
if not bot_token or not chat_id:
    print("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
    sys.exit(1)

print(f"Bot: {bot_token[:10]}...")
print(f"Chat: {chat_id}")

with open("output/daily_digest.json", "r", encoding="utf-8") as f:
    data = json.load(f)
items = data["items"]

now = datetime.now(timezone.utc)

def parse_date(pub):
    if not pub: return None
    try:
        if "T" in str(pub):
            return datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        else:
            from email.utils import parsedate_to_datetime
            return parsedate_to_datetime(pub)
    except: return None

def relative_time(pub_dt):
    if not pub_dt: return ""
    diff = now - pub_dt
    hours = int(diff.total_seconds() / 3600)
    if hours < 1: return "همین الان"
    elif hours < 24: return f"{hours} ساعت پیش"
    else: return f"{int(hours/24)} روز پیش"

def gregorian_to_jalali(gy, gm, gd):
    g_d_m = [0,31,59,90,120,151,181,212,243,273,304,334]
    gy2 = gy + 1 if gm > 2 else gy
    days = 355666 + (365*gy) + ((gy2+3)//4) - ((gy2+99)//100) + ((gy2+399)//400) + gd + g_d_m[gm-1]
    jy = -1595 + (33*(days//12053))
    days %= 12053
    jy += 4*(days//1461)
    days %= 1461
    if days > 365:
        jy += (days-1)//365
        days = (days-1)%365
    if days < 186:
        jm = 1 + (days//31)
        jd = 1 + (days%31)
    else:
        jm = 7 + ((days-186)//30)
        jd = 1 + ((days-186)%30)
    return jy, jm, jd

MONTHS_FA = ["","فروردین","اردیبهشت","خرداد","تیر","مرداد","شهریور","مهر","آبان","آذر","دی","بهمن","اسفند"]

def fmt_jalali(dt):
    if not dt: return ""
    jy,jm,jd = gregorian_to_jalali(dt.year, dt.month, dt.day)
    return f"{jd} {MONTHS_FA[jm]} {jy}"

RLM = "\u200F"
LRM = "\u200E"

# Header
stats = data["stats"]
header = f"""<b>خبرنامه هوش مصنوعی</b>
{data['date']} | {fmt_jalali(now)}
━━━━━━━━━━━━━━━━
{stats['total_sources']} منبع | {len(items)} خبر برتر"""

r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
    json={"chat_id":chat_id,"text":header,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=10)
print("header sent" if r.json().get("ok") else f"header err: {r.json()}")
time.sleep(0.5)

# Items
sent = 0
for item in items[:12]:
    score = item.get("importance_score", 0)
    headline = item.get("headline_fa", item.get("title", ""))
    summary = item.get("summary_fa", "")
    why = item.get("why_it_matters", "")
    url = item.get("url", "")
    src = item.get("source", "")
    viral = item.get("viral_potential", "")
    opp = item.get("opportunity", "no")
    tags = item.get("tags", [])

    pub_dt = parse_date(item.get("published_at", ""))
    date_fa = fmt_jalali(pub_dt)
    rel = relative_time(pub_dt)

    viral_icon = {"high":"🔥","medium":"📈","low":"📌"}.get(viral, "")
    si = "🔴" if score >= 9 else "🟡" if score >= 8 else "🟢"

    opp_line = ""
    if str(opp).startswith("yes"):
        ot = str(opp).replace("yes — ","").replace("yes - ","").replace("yes","").strip()
        if ot: opp_line = f"\n⚡ <b>فرصت:</b> <i>{LRM}{ot}{RLM}</i>"

    tag_line = ""
    if tags:
        tag_line = "\n🏷 " + " ".join("#"+t.replace(" ","_") for t in tags[:5])

    date_str = f"📅 {date_fa}" + (f" ({rel})" if rel else "")

    text = f"""{si} <b>{score}/۱۰</b> {viral_icon} ━ {LRM}{src}{RLM}
{date_str}

<b>{headline}</b>

{summary}

💡 <i>{why}</i>{opp_line}{tag_line}

🔗 <a href="{url}">منبع</a>"""

    try:
        r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id":chat_id,"text":text,"parse_mode":"HTML","disable_web_page_preview":True}, timeout=10)
        if r.json().get("ok"):
            sent += 1
            print(f"  ok [{score}] {headline[:40]}")
        else:
            fb = f"{si} {score}/10 — {src}\n{date_fa}\n{headline}\n{summary}\n{url}"
            r2 = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id":chat_id,"text":fb}, timeout=10)
            if r2.json().get("ok"):
                sent += 1
                print(f"  ok [{score}] {headline[:40]} (plain)")
            else:
                print(f"  err [{score}] {r.json().get('description','?')}")
    except Exception as e:
        print(f"  err {e}")
    time.sleep(0.5)

footer = f"━━━━━━━━━━━━━━━━\n📬 {sent} خبر ارسال شد"
requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
    json={"chat_id":chat_id,"text":footer,"parse_mode":"HTML"}, timeout=10)
print(f"\n{sent} sent to telegram")
