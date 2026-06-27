"""
AI Daily Digest Collector — نسخه کامل با OpenRouter
مدل انتخابی: DeepSeek R1 0528 (استدلال ۱۰/۱۰ — فارسی ۹/۱۰ — $0.50/$2.15 per 1M)
هزینه واقعی: ~$0.05 برای ۳۰۰ آیتم/روز (بسیار پایین‌تر از بودجه $0.50)
"""

import feedparser
import requests
import json
import time
import os
import hashlib
import sys
import threading
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

# حل مشکل انکودینگ ترمینال در ویندوز
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ─────────────────────────────────────────
# تنظیمات
# ─────────────────────────────────────────

# تنظیمات پیش‌فرض
OUTPUT_DIR = "output"
MAX_ITEMS_TO_PROCESS = 300
MIN_SCORE_TO_SHOW = 6.0

def load_config():
    global OUTPUT_DIR, MAX_ITEMS_TO_PROCESS, MIN_SCORE_TO_SHOW
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                OUTPUT_DIR = cfg.get("output_dir", OUTPUT_DIR)
                MAX_ITEMS_TO_PROCESS = cfg.get("max_items_to_process", MAX_ITEMS_TO_PROCESS)
                MIN_SCORE_TO_SHOW = cfg.get("min_score_to_show", MIN_SCORE_TO_SHOW)
                return cfg
        except Exception as e:
            print(f"⚠️ خطا در خواندن config.json: {e}")
    return {}

class BaseProvider:
    def __init__(self, config):
        self.config = config
        api_key_env = config.get("api_key_env", "")
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        if not self.api_key:
            self.api_key = config.get("api_key", "")
        self.model_id = config.get("model_id", "")
        self.temperature = config.get("temperature", 0.3)
        self.endpoint_url = config.get("endpoint_url", "")

    def analyze(self, item, prompt):
        raise NotImplementedError

class OpenAICompatibleProvider(BaseProvider):
    def analyze(self, item, prompt):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self.endpoint_url:
            headers["HTTP-Referer"] = "https://github.com/ai-digest"
            headers["X-Title"] = "AI Daily Digest"

        payload = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": 1500,
        }
        try:
            r = requests.post(self.endpoint_url, headers=headers, json=payload, timeout=40)
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return self._parse_json(text)
        except Exception as e:
            print(f"    ⚠️ خطا در پرووایدر OpenAI-Compatible: {e}")
            return None

    def _parse_json(self, text):
        import re
        # حذف تگ‌های <think>...</think> مدل‌های Reasoning مثل DeepSeek R1
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # حذف markdown code block
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        # یافتن اولین { تا آخرین } برای استخراج JSON معتبر
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)

class GeminiProvider(BaseProvider):
    def analyze(self, item, prompt):
        url = f"{self.endpoint_url}/{self.model_id}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature
            }
        }
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return self._parse_json(text)
        except Exception as e:
            print(f"    ⚠️ خطا در پرووایدر Gemini: {e}")
            return None

    def _parse_json(self, text):
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end+1]
        return json.loads(text)


def get_provider(config_data):
    active = config_data.get("active_provider", "openrouter")
    prov_cfg = config_data.get("providers", {}).get(active)
    if not prov_cfg:
        print(f"⚠️ پرووایدر '{active}' یافت نشد. از تنظیمات پیش‌فرض OpenRouter استفاده می‌شود.")
        prov_cfg = {
            "type": "openai_compatible",
            "endpoint_url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key_env": "OPENROUTER_API_KEY",
            "model_id": "deepseek/deepseek-r1-0528",
            "temperature": 0.3
        }
    
    p_type = prov_cfg.get("type", "openai_compatible")
    if p_type == "openai_compatible":
        return OpenAICompatibleProvider(prov_cfg)
    elif p_type == "gemini":
        return GeminiProvider(prov_cfg)
    else:
        print(f"⚠️ نوع پرووایدر ناشناخته: {p_type}. استفاده از OpenAI-Compatible.")
        return OpenAICompatibleProvider(prov_cfg)


# ─────────────────────────────────────────
# ۱. رسانه‌های فناوری — RSS
# ─────────────────────────────────────────
RSS_SOURCES = [
    # ── رسانه‌های اصلی فناوری ──
    {"name": "MIT Technology Review",  "url": "https://www.technologyreview.com/feed/",                                  "category": "research"},
    {"name": "IEEE Spectrum AI",       "url": "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss",      "category": "research"},
    {"name": "VentureBeat AI",         "url": "https://venturebeat.com/category/ai/feed/",                              "category": "startup"},
    {"name": "The Verge AI",           "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",      "category": "viral"},
    {"name": "TechCrunch AI",          "url": "https://techcrunch.com/category/artificial-intelligence/feed/",          "category": "startup"},
    {"name": "Wired AI",               "url": "https://www.wired.com/feed/category/artificial-intelligence/latest/rss", "category": "viral"},

    # ── وبلاگ‌های رسمی آزمایشگاه‌ها ──
    {"name": "Google AI Blog",         "url": "https://research.google/blog/feed/",                                     "category": "model"},
    {"name": "DeepMind Blog",          "url": "https://deepmind.google/blog/rss.xml",                                   "category": "model"},
    {"name": "OpenAI Blog",            "url": "https://openai.com/blog/rss.xml",                                        "category": "model"},
    {"name": "Anthropic News",         "url": "https://www.anthropic.com/news/rss.xml",                                  "category": "model"},
    {"name": "Hugging Face Blog",      "url": "https://huggingface.co/blog/feed.xml",                                   "category": "model"},
    {"name": "Meta AI Blog",           "url": "https://ai.meta.com/blog/feed/",                                         "category": "model"},
    {"name": "Microsoft AI Blog",      "url": "https://blogs.microsoft.com/ai/feed/",                                   "category": "model"},
    {"name": "Mistral AI Blog",        "url": "https://mistral.ai/news/rss.xml",                                        "category": "model"},
    {"name": "Cohere Blog",            "url": "https://cohere.com/blog/rss.xml",                                        "category": "model"},

    # ── arXiv — تحقیقات علمی (سه حوزه اصلی) ──
    {"name": "ArXiv CS.AI",            "url": "https://rss.arxiv.org/rss/cs.AI",                                       "category": "research"},
    {"name": "ArXiv CS.LG",            "url": "https://rss.arxiv.org/rss/cs.LG",                                       "category": "research"},
    {"name": "ArXiv CS.CL",            "url": "https://rss.arxiv.org/rss/cs.CL",                                       "category": "research"},

    # ── هکاتون و رویداد ──
    {"name": "Lablab.ai Blog",         "url": "https://lablab.ai/blog/rss.xml",                                         "category": "hackathon"},
    {"name": "AngelHack Blog",         "url": "https://angelhack.com/blog/feed/",                                       "category": "hackathon"},
]

# ─────────────────────────────────────────
# ۲. خبرنامه‌های ایمیلی — اسکرپ وب + RSS
#    (اکثر RSS عمومی ندارند، از آرشیو وب اسکرپ می‌شوند)
# ─────────────────────────────────────────
NEWSLETTER_SCRAPE = [
    {
        "name": "TLDR AI",
        "url": "https://tldr.tech/ai",
        "selector": "article, .issue-item",
        "title_sel": "h3, h2",
        "text_sel": "p",
        "category": "research",
        "limit": 8,
        "note": "۱.۲۵M مشترک — خلاصه فشرده روزانه برای مهندسان"
    },
    {
        "name": "The Rundown AI",
        "url": "https://www.therundown.ai/",
        "selector": ".post-preview, article, .story",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "viral",
        "limit": 6,
        "note": "۲M+ مشترک — آموزش کاربردی همان روز انتشار مدل"
    },
    {
        "name": "Ben's Bites",
        "url": "https://bensbites.beehiiv.com/",
        "selector": ".post-preview-content, article",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "startup",
        "limit": 5,
        "note": "۱۲۰K مشترک — تمرکز بنیان‌گذاران و استراتژی محصول"
    },
    {
        "name": "Superhuman AI",
        "url": "https://www.superhumanai.com/",
        "selector": "article, .post",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "tool",
        "limit": 5,
        "note": "۱M+ مشترک — پرامپت و ابزار کاربردی"
    },
    {
        "name": "Import AI",
        "url": "https://importai.substack.com/",
        "selector": ".post-preview, article",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "research",
        "limit": 3,
        "note": "۵۰K+ مشترک — جک کلارک (هم‌بنیان‌گذار Anthropic) — سیاست و ایمنی"
    },
    {
        "name": "Interconnects",
        "url": "https://www.interconnects.ai/",
        "selector": "article, .post-preview",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "research",
        "limit": 3,
        "note": "۶۰K+ مشترک — تحلیل عمیق RLHF و پس‌آموزش"
    },
    {
        "name": "AI Supremacy (Substack)",
        "url": "https://aisupremacy.substack.com/",
        "selector": "article, .post-preview",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "research",
        "limit": 3,
        "note": "تحلیل معماری مدل‌های پیشرفته — مخاطب فنی"
    },
    {
        "name": "The Batch (DeepLearning.AI)",
        "url": "https://www.deeplearning.ai/the-batch/",
        "selector": "article, .post-card",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "research",
        "limit": 4,
        "note": "۲۰۰K+ مشترک — اندرو نگ — سنتز تکنیکال هفتگی"
    },
    {
        "name": "Techpresso",
        "url": "https://dupple.com/techpresso",
        "selector": "article, .story",
        "title_sel": "h2, h3",
        "text_sel": "p",
        "category": "viral",
        "limit": 4,
        "note": "۵۰۰K+ مشترک — فیلتر شایعات، بنچمارک‌های مهجور"
    },
]

# ─────────────────────────────────────────
# ۳. Reddit — JSON API عمومی (بدون نیاز به API key)
# ─────────────────────────────────────────
REDDIT_SOURCES = [
    # آکادمیک و تحقیقاتی
    {"subreddit": "MachineLearning",  "limit": 15, "sort": "hot"},
    {"subreddit": "LocalLLaMA",       "limit": 15, "sort": "hot"},
    {"subreddit": "artificial",       "limit": 10, "sort": "hot"},
    {"subreddit": "singularity",      "limit": 8,  "sort": "hot"},
    # کاربران عمومی و وایرال
    {"subreddit": "ChatGPT",          "limit": 8,  "sort": "top", "time": "day"},
    {"subreddit": "ChatGPTPro",       "limit": 5,  "sort": "hot"},
    # توسعه‌دهندگان
    {"subreddit": "VibeCoding",       "limit": 6,  "sort": "hot"},
    {"subreddit": "ClaudeAI",         "limit": 5,  "sort": "hot"},
    {"subreddit": "ChatGPTCoding",    "limit": 5,  "sort": "hot"},
]

# ─────────────────────────────────────────
# ۴. منابع تخصصی — API / اسکرپ مستقیم
# ─────────────────────────────────────────
SPECIAL_SOURCES = [
    # ارزیابی مدل‌ها
    {
        "name": "LMSYS Arena Leaderboard",
        "type": "scrape",
        "url": "https://lmarena.ai/",
        "category": "model",
        "note": "استاندارد طلایی ارزیابی مبتنی بر ترجیح انسانی — سیستم Elo"
    },
    {
        "name": "Open LLM Leaderboard (HF)",
        "type": "scrape",
        "url": "https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard",
        "category": "model",
        "note": "رتبه‌بندی بلادرنگ مدل‌های متن‌باز"
    },
    {
        "name": "LiveBench",
        "type": "scrape",
        "url": "https://livebench.ai/",
        "category": "model",
        "note": "بنچمارک بدون آلودگی داده — متن‌باز (آوریل ۲۰۲۵)"
    },

    # مدل‌های ترند
    {
        "name": "HF Trending Models",
        "type": "hf_trending",
        "url": "https://huggingface.co/api/models?sort=trending&limit=10&full=false",
        "category": "model"
    },
    {
        "name": "HF Daily Papers",
        "type": "rss",
        "url": "https://huggingface.co/papers.rss",
        "category": "research",
        "note": "مقالات روز که جامعه HF بیشترین توجه به آن‌ها دارد"
    },

    # تحقیقات
    {
        "name": "Papers With Code",
        "type": "rss",
        "url": "https://paperswithcode.com/latest.rss",
        "category": "research",
        "note": "مقالات با کد — SOTA جدید"
    },

    # هکاتون
    {
        "name": "Lablab.ai Events",
        "type": "scrape_lablab",
        "url": "https://lablab.ai/event",
        "category": "hackathon",
        "note": "هکاتون‌های فعال با جوایز نقدی"
    },

    # GitHub — سیگنال اولیه ترندها
    {
        "name": "GitHub Trending AI",
        "type": "github_trending",
        "url": "https://github.com/trending?l=python&since=daily",
        "category": "tool",
        "note": "ریپوهای ترند روزانه پایتون — اولین سیگنال ابزارهای جدید"
    },

    # GitHub Awesome Lists — کیوریت‌شده
    {
        "name": "awesome-evals (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/benchflow-ai/awesome-evals/commits",
        "category": "research",
        "note": "۴۴۳+ منبع ارزیابی عامل‌های AI — به‌روزرسانی چند بار در هفته"
    },
    {
        "name": "awesome-opensource-ai (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/alvinreal/awesome-opensource-ai/commits",
        "category": "tool",
        "note": "پروژه‌های متن‌باز AI — به‌روزرسانی روزانه"
    },
    {
        "name": "awesome-ai-agents-2026 (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/caramaschiHG/awesome-ai-agents-2026/commits",
        "category": "tool",
        "note": "۳۰۰+ فریم‌ورک و ابزار Agentic AI"
    },

    # منابع رایگان API
    {
        "name": "OpenRouter Free Models",
        "type": "openrouter_free",
        "url": "https://openrouter.ai/api/v1/models",
        "category": "free-resource",
        "note": "لیست مدل‌های رایگان OpenRouter — آپدیت روزانه"
    },
]


# ─────────────────────────────────────────
# توابع جمع‌آوری
# ─────────────────────────────────────────

def fetch_rss(source):
    """خواندن فید RSS"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
        resp = requests.get(source["url"], headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        items = []
        for entry in feed.entries[:12]:
            summary = entry.get("summary", entry.get("description", ""))
            clean_text = BeautifulSoup(summary, "html.parser").get_text(separator=" ", strip=True)[:800]
            items.append({
                "id": hashlib.md5((entry.get("link", "") + entry.get("title", "")).encode()).hexdigest()[:10],
                "source": source["name"],
                "base_category": source["category"],
                "title": entry.get("title", "")[:200],
                "url": entry.get("link", ""),
                "summary_raw": clean_text,
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
            })
        print(f"  ✅ {source['name']}: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ {source['name']}: {e}")
        return []



def fetch_newsletter_scrape(cfg):
    """اسکرپ صفحه وب خبرنامه‌ها"""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)",
        "Accept-Language": "en-US,en;q=0.9"
    }
    try:
        r = requests.get(cfg["url"], headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        containers = soup.select(cfg["selector"])[:cfg["limit"]]
        items = []
        for container in containers:
            title_el = container.select_one(cfg["title_sel"])
            text_el  = container.select_one(cfg["text_sel"])
            link_el  = container.select_one("a[href]")

            title = title_el.get_text(strip=True) if title_el else ""
            text  = text_el.get_text(strip=True)[:600] if text_el else ""
            url   = link_el["href"] if link_el else cfg["url"]
            if not url.startswith("http"):
                base = "/".join(cfg["url"].split("/")[:3])
                url = base + url

            if not title or len(title) < 5:
                continue

            items.append({
                "id": hashlib.md5(url.encode()).hexdigest()[:10],
                "source": cfg["name"],
                "base_category": cfg["category"],
                "title": title[:200],
                "url": url,
                "summary_raw": text,
                "published_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"  ✅ {cfg['name']}: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ {cfg['name']}: {e}")
        return []


def fetch_reddit(config):
    """خواندن Reddit از طریق JSON عمومی"""
    url = f"https://www.reddit.com/r/{config['subreddit']}/{config['sort']}.json?limit={config['limit']}"
    if config.get("time"):
        url += f"&t={config['time']}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        items = []
        for post in data["data"]["children"]:
            p = post["data"]
            if p.get("is_self"):
                text = (p.get("selftext", "") or "")[:600]
            else:
                text = p.get("title", "")
            items.append({
                "id": hashlib.md5(p.get("url", p["title"]).encode()).hexdigest()[:10],
                "source": f"r/{config['subreddit']}",
                "base_category": "viral",
                "title": p["title"][:200],
                "url": f"https://reddit.com{p['permalink']}",
                "summary_raw": text,
                "score_reddit": p.get("score", 0),
                "published_at": datetime.fromtimestamp(p["created_utc"], tz=timezone.utc).isoformat(),
            })
        time.sleep(2)
        print(f"  ✅ r/{config['subreddit']}: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ r/{config['subreddit']}: {e}")
        return []


def fetch_hf_trending():
    """مدل‌های ترند Hugging Face"""
    try:
        r = requests.get(
            "https://huggingface.co/api/models?sort=trending&limit=10&full=false",
            timeout=10
        )
        models = r.json()
        if not isinstance(models, list):
            print(f"  ❌ HF Trending: Expected list response, got {type(models)}")
            return []
        items = []
        for m in models:
            name = m.get("modelId", m.get("id", ""))
            items.append({
                "id": hashlib.md5(name.encode()).hexdigest()[:10],
                "source": "HF Trending Models",
                "base_category": "model",
                "title": f"مدل ترند: {name}",
                "url": f"https://huggingface.co/{name}",
                "summary_raw": f"likes: {m.get('likes',0)} | downloads: {m.get('downloads',0)} | pipeline: {m.get('pipeline_tag','')}",
                "published_at": m.get("lastModified", datetime.now(timezone.utc).isoformat()),
            })
        print(f"  ✅ HF Trending: {len(items)} مدل")
        return items
    except Exception as e:
        print(f"  ❌ HF Trending: {e}")
        return []


def fetch_papers_with_code():
    """جدیدترین مقالات Papers With Code"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
        resp = requests.get("https://paperswithcode.com/latest.rss", headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        items = []
        for entry in feed.entries[:8]:
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:600]
            items.append({
                "id": hashlib.md5(entry.get("link", "").encode()).hexdigest()[:10],
                "source": "Papers With Code",
                "base_category": "research",
                "title": entry.get("title", "")[:200],
                "url": entry.get("link", ""),
                "summary_raw": summary,
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
            })
        print(f"  ✅ Papers With Code: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ Papers With Code: {e}")
        return []



def fetch_github_trending():
    """ریپوهای ترند روزانه Python در GitHub — سیگنال اولیه ابزارهای جدید"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
    try:
        r = requests.get("https://github.com/trending/python?since=daily", headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        repos = soup.select("article.Box-row")[:10]
        for repo in repos:
            name_el = repo.select_one("h2 a")
            desc_el = repo.select_one("p")
            stars_el = repo.select_one("a[href$='/stargazers']")
            if not name_el:
                continue
            name = name_el.get_text(strip=True).replace("\n", "").replace(" ", "")
            desc = desc_el.get_text(strip=True) if desc_el else ""
            stars = stars_el.get_text(strip=True) if stars_el else "?"
            url = "https://github.com" + name_el["href"].strip()

            # فقط ریپوهای مرتبط با AI
            ai_keywords = ["llm", "ai", "gpt", "model", "agent", "ml", "neural",
                           "transformer", "diffusion", "rag", "embed", "inference",
                           "openai", "claude", "gemini", "llama", "deepseek"]
            combined = (name + " " + desc).lower()
            if not any(kw in combined for kw in ai_keywords):
                continue

            items.append({
                "id": hashlib.md5(url.encode()).hexdigest()[:10],
                "source": "GitHub Trending AI",
                "base_category": "tool",
                "title": f"GitHub Trending: {name} — ⭐{stars}",
                "url": url,
                "summary_raw": desc,
                "published_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"  ✅ GitHub Trending AI: {len(items)} ریپو")
        return items
    except Exception as e:
        print(f"  ❌ GitHub Trending: {e}")
        return []


def fetch_github_awesome_commits(cfg):
    """آخرین تغییرات در لیست‌های Awesome GitHub"""
    try:
        r = requests.get(cfg["url"] + "?per_page=5",
                         headers={"Accept": "application/vnd.github.v3+json",
                                  "User-Agent": "AI-Digest-Bot/1.0"},
                         timeout=10)
        commits = r.json()
        if not isinstance(commits, list):
            return []
        items = []
        repo_name = cfg["url"].split("/repos/")[1].split("/commits")[0]
        for commit in commits[:5]:
            msg = commit.get("commit", {}).get("message", "")[:150]
            sha = commit.get("sha", "")[:7]
            url = f"https://github.com/{repo_name}/commit/{sha}"
            if len(msg) < 10:
                continue
            items.append({
                "id": hashlib.md5(sha.encode()).hexdigest()[:10],
                "source": cfg["name"],
                "base_category": cfg["category"],
                "title": f"[{repo_name.split('/')[-1]}] {msg}",
                "url": url,
                "summary_raw": cfg.get("note", "") + " — " + msg,
                "published_at": commit.get("commit", {}).get("committer", {}).get("date",
                                datetime.now(timezone.utc).isoformat()),
            })
        print(f"  ✅ {cfg['name']}: {len(items)} کامیت جدید")
        return items
    except Exception as e:
        print(f"  ❌ {cfg['name']}: {e}")
        return []


def fetch_openrouter_free_models():
    """لیست مدل‌های رایگان OpenRouter از API رسمی"""
    try:
        r = requests.get("https://openrouter.ai/api/v1/models",
                         headers={"User-Agent": "AI-Digest-Bot/1.0"}, timeout=10)
        data = r.json()
        models = data.get("data", [])
        free_models = [m for m in models if m.get("pricing", {}).get("prompt") == "0"]
        if not free_models:
            return []
        names = ", ".join([m.get("name", m.get("id", "")) for m in free_models[:10]])
        items = [{
            "id": hashlib.md5("openrouter_free_today".encode()).hexdigest()[:10],
            "source": "OpenRouter Free Models",
            "base_category": "free-resource",
            "title": f"مدل‌های رایگان OpenRouter امروز: {len(free_models)} مدل",
            "url": "https://openrouter.ai/models?q=free",
            "summary_raw": f"مدل‌های رایگان: {names}",
            "published_at": datetime.now(timezone.utc).isoformat(),
        }]
        print(f"  ✅ OpenRouter Free: {len(free_models)} مدل رایگان")
        return items
    except Exception as e:
        print(f"  ❌ OpenRouter Free: {e}")
        return []


def fetch_hf_daily_papers():
    """مقالات روز Hugging Face"""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
        resp = requests.get("https://huggingface.co/papers.rss", headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        items = []
        for entry in feed.entries[:8]:
            summary = BeautifulSoup(entry.get("summary", ""), "html.parser").get_text()[:600]
            items.append({
                "id": hashlib.md5(entry.get("link", "").encode()).hexdigest()[:10],
                "source": "HF Daily Papers",
                "base_category": "research",
                "title": entry.get("title", "")[:200],
                "url": entry.get("link", ""),
                "summary_raw": summary,
                "published_at": entry.get("published", datetime.now(timezone.utc).isoformat()),
            })
        print(f"  ✅ HF Daily Papers: {len(items)} مقاله")
        return items
    except Exception as e:
        print(f"  ❌ HF Daily Papers: {e}")
        return []



def fetch_lablab_events():
    """هکاتون‌های فعال Lablab"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
    try:
        r = requests.get("https://lablab.ai/event", headers=headers, timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        items = []
        # یافتن کارت‌های رویداد
        cards = soup.select("a[href*='/event/']")[:8]
        seen_urls = set()
        for card in cards:
            url = "https://lablab.ai" + card["href"] if card["href"].startswith("/") else card["href"]
            if url in seen_urls:
                continue
            seen_urls.add(url)
            title_el = card.select_one("h2, h3, .title, strong")
            title = title_el.get_text(strip=True) if title_el else card.get_text(strip=True)[:80]
            if not title or len(title) < 5:
                continue
            items.append({
                "id": hashlib.md5(url.encode()).hexdigest()[:10],
                "source": "Lablab.ai Events",
                "base_category": "hackathon",
                "title": f"هکاتون: {title[:150]}",
                "url": url,
                "summary_raw": "هکاتون فعال روی پلتفرم Lablab.ai",
                "published_at": datetime.now(timezone.utc).isoformat(),
            })
        print(f"  ✅ Lablab Events: {len(items)} هکاتون فعال")
        return items
    except Exception as e:
        print(f"  ❌ Lablab Events: {e}")
        return []


# ─────────────────────────────────────────
# جمع‌آوری کل
# ─────────────────────────────────────────

def collect_all():
    print("\n📡 مرحله ۱: جمع‌آوری داده‌ها\n")
    all_items = []

    print("── RSS رسانه‌های رسمی و آزمایشگاه‌ها ──")
    for source in RSS_SOURCES:
        all_items.extend(fetch_rss(source))
        time.sleep(0.5)

    print("\n── خبرنامه‌های تخصصی (اسکرپ) ──")
    for cfg in NEWSLETTER_SCRAPE:
        all_items.extend(fetch_newsletter_scrape(cfg))
        time.sleep(1.5)

    print("\n── Reddit ──")
    for cfg in REDDIT_SOURCES:
        all_items.extend(fetch_reddit(cfg))

    print("\n── منابع تخصصی ──")
    all_items.extend(fetch_hf_trending())
    all_items.extend(fetch_hf_daily_papers())
    all_items.extend(fetch_papers_with_code())
    all_items.extend(fetch_lablab_events())

    print("\n── GitHub (ترند + Awesome Lists) ──")
    all_items.extend(fetch_github_trending())
    for src in SPECIAL_SOURCES:
        if src.get("type") == "github_commits":
            all_items.extend(fetch_github_awesome_commits(src))
            time.sleep(1)

    print("\n── OpenRouter Free Models ──")
    all_items.extend(fetch_openrouter_free_models())

    # حذف تکراری
    seen = set()
    unique = []
    for item in all_items:
        if item["id"] not in seen:
            seen.add(item["id"])
            unique.append(item)

    total_sources = len(RSS_SOURCES) + len(NEWSLETTER_SCRAPE) + len(REDDIT_SOURCES) + 7
    print(f"\n📊 {total_sources} منبع فعال — {len(all_items)} آیتم → {len(unique)} آیتم یکتا")
    return unique


# ─────────────────────────────────────────
# پردازش Gemini
# ─────────────────────────────────────────

PRIORITY_KEYWORDS = [
    "release", "launch", "open-source", "free", "benchmark", "hackathon",
    "llm", "gpt", "claude", "gemini", "llama", "deepseek", "qwen", "agent",
    "model", "api", "paper", "funding", "billion", "state-of-the-art",
    "trending", "sota", "fine-tun", "quantiz", "rlhf", "multimodal",
    "reasoning", "agentic", "autonomous", "breakthrough", "leaderboard"
]

def priority_score(item):
    score = sum(1 for kw in PRIORITY_KEYWORDS if kw in item["title"].lower())
    if item.get("score_reddit", 0) > 500: score += 3
    if item["base_category"] in ["model", "hackathon"]: score += 2
    if item["base_category"] == "research": score += 1
    return score


def analyze_item(item, provider):
    """تحلیل هر آیتم با پرووایدر هوش مصنوعی فعال"""
    prompt = f"""تو یه ویراستار تخصصی هوش مصنوعی هستی که برای مخاطبان فارسی‌زبان محتوا می‌سازی.
این خبر/پست رو بررسی کن:

منبع: {item['source']}
عنوان: {item['title']}
محتوا: {item.get('summary_raw','')[:500]}

خروجی رو دقیقاً به این فرمت JSON بده — هیچ توضیح اضافه‌ای نده:
{{
  "importance_score": <عدد 1 تا 10>,
  "categories": <آرایه از: "model", "tool", "research", "startup", "hackathon", "viral", "free-resource">,
  "headline_fa": "<عنوان جذاب فارسی حداکثر ۱۲ کلمه>",
  "summary_fa": "<خلاصه ۲ جمله فارسی ساده و کاربردی>",
  "why_it_matters": "<یه جمله فارسی: چرا مهمه؟>",
  "opportunity": "<yes یا no — اگر yes: توضیح کوتاه فرصت عملی/درآمدی>",
  "viral_potential": "<high یا medium یا low>",
  "tags": <آرایه تگ‌های انگلیسی کوتاه>
}}"""
    return provider.analyze(item, prompt)


def process_items(items, provider):
    print(f"\n🧠 مرحله ۲: پردازش با {provider.model_id}\n")

    sorted_items = sorted(items, key=priority_score, reverse=True)
    to_process = sorted_items[:MAX_ITEMS_TO_PROCESS]

    # تخمین هزینه بر اساس مدل فعال
    cost_in = 0.50
    cost_out = 2.15
    if "gpt-4o-mini" in provider.model_id:
        cost_in = 0.150
        cost_out = 0.600
    elif "gemini" in provider.config.get("type", "") or "ollama" in provider.config.get("type", "") or "localhost" in provider.endpoint_url:
        cost_in = 0.0
        cost_out = 0.0

    cost_est = len(to_process) * (450 * cost_in + 200 * cost_out) / 1_000_000
    print(f"  پردازش {len(to_process)} آیتم از {len(items)} کل")
    print(f"  تخمین هزینه: ${cost_est:.4f}\n")

    results = []
    for i, item in enumerate(to_process):
        print(f"  [{i+1}/{len(to_process)}] {item['title'][:55]}...")
        analysis = analyze_item(item, provider)
        if analysis:
            results.append({**item, **analysis})
        time.sleep(0.8)

    total_cost = len(results) * (450 * cost_in + 200 * cost_out) / 1_000_000
    print(f"\n  ✅ پردازش کامل — هزینه واقعی: ${total_cost:.4f}")
    return results


# ─────────────────────────────────────────
# ذخیره
# ─────────────────────────────────────────

def save_results(items, total_fetched):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/archive", exist_ok=True)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    filtered = [
        item for item in items
        if (item.get("importance_score", 0) >= MIN_SCORE_TO_SHOW
            or item.get("viral_potential") == "high"
            or str(item.get("opportunity", "no")).startswith("yes"))
    ]
    filtered.sort(key=lambda x: x.get("importance_score", 0), reverse=True)

    output = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_sources": len(RSS_SOURCES) + len(NEWSLETTER_SCRAPE) + len(REDDIT_SOURCES) + 3,
            "total_fetched": total_fetched,
            "total_processed": len(items),
            "total_displayed": len(filtered),
            "top_score": filtered[0]["importance_score"] if filtered else 0,
            "high_viral_count": sum(1 for x in filtered if x.get("viral_potential") == "high"),
            "opportunities_count": sum(1 for x in filtered if str(x.get("opportunity","no")).startswith("yes")),
        },
        "items": filtered
    }

    for path in [f"{OUTPUT_DIR}/daily_digest.json", f"{OUTPUT_DIR}/archive/{today}.json"]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(filtered)} آیتم ذخیره شد")

    # آمار منابع
    sources_count = {}
    for item in filtered:
        s = item["source"]
        sources_count[s] = sources_count.get(s, 0) + 1
    print("\n📊 توزیع منابع در خروجی نهایی:")
    for src, cnt in sorted(sources_count.items(), key=lambda x: -x[1]):
        print(f"  {src}: {cnt}")

    return output


# وضعیت جهانی اجرای وظیفه واکشی
_fetch_job = {"status": "idle", "message": "", "total_displayed": 0}
_fetch_lock = threading.Lock()

def _run_fetch_job():
    """اجرای جمع‌آوری اخبار در یک Thread جداگانه"""
    global _fetch_job
    try:
        config_data = load_config()
        provider = get_provider(config_data)
        raw = collect_all()
        processed = process_items(raw, provider)
        save_results(processed, len(raw))
        displayed = sum(1 for item in processed if (
            item.get("importance_score", 0) >= MIN_SCORE_TO_SHOW
            or item.get("viral_potential") == "high"
            or str(item.get("opportunity", "no")).startswith("yes")
        ))
        with _fetch_lock:
            _fetch_job = {
                "status": "success",
                "message": f"{len(processed)} خبر پردازش و به‌روزرسانی شد.",
                "total_displayed": displayed
            }
        print("\n✅ واکشی پس‌زمینه با موفقیت انجام شد.")
    except Exception as e:
        print(f"\n❌ خطا در واکشی پس‌زمینه: {e}")
        with _fetch_lock:
            _fetch_job = {"status": "error", "message": str(e), "total_displayed": 0}


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        global _fetch_job
        parsed = urlparse(self.path)
        if parsed.path == "/api/fetch":
            with _fetch_lock:
                current_status = _fetch_job["status"]
            
            if current_status == "running":
                # اگر قبلاً در حال اجراست، خطا بده
                self.send_response(409)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                res = {"status": "running", "message": "واکشی اخبار از قبل در حال اجراست. لطفاً صبر کنید."}
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
                return
            
            # ریست وضعیت و شروع thread جدید
            with _fetch_lock:
                _fetch_job = {"status": "running", "message": "در حال جمع‌آوری اخبار...", "total_displayed": 0}
            
            t = threading.Thread(target=_run_fetch_job, daemon=True)
            t.start()
            print("\n🔄 thread واکشی اخبار شروع شد...")
            
            # پاسخ فوری به مرورگر
            self.send_response(202)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            res = {"status": "running", "message": "واکشی اخبار شروع شد. وضعیت را از /api/status دنبال کنید."}
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        global _fetch_job
        parsed = urlparse(self.path)
        clean_path = parsed.path
        
        # مسیر وضعیت واکشی
        if clean_path == "/api/status":
            with _fetch_lock:
                state = dict(_fetch_job)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(state, ensure_ascii=False).encode('utf-8'))
            return
        
        # هدایت مسیر روت به dashboard.html
        if clean_path in ["/", "/index.html"]:
            try:
                with open("dashboard.html", "rb") as f:
                    content = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_error(404, f"dashboard.html index file not found: {e}")
            return
            
        # هدایت مسیر daily_digest.json به فایل واقعی در پوشه output
        if clean_path == "/daily_digest.json":
            target_path = os.path.join(OUTPUT_DIR, "daily_digest.json")
            if os.path.exists(target_path):
                try:
                    with open(target_path, "rb") as f:
                        content = f.read()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(content)
                except Exception as e:
                    self.send_error(500, f"Error reading daily_digest.json: {e}")
            else:
                self.send_response(404)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                res = {"error": "فایل داده‌ها یافت نشد. لطفاً ابتدا اخبار را واکشی کنید."}
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode('utf-8'))
            return
            
        # سایر فایل‌های استاتیک
        super().do_GET()

def start_server(port=8000):
    server_address = ('', port)
    httpd = ThreadingHTTPServer(server_address, DashboardHandler)
    print("=" * 65)
    print(f"[SERVER] Dashboard ready: http://localhost:{port}")
    print("Press Ctrl+C to stop.")
    print("=" * 65)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Stopped.")
        sys.exit(0)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        start_server()
        return

    print("=" * 65)
    print("🤖 AI Daily Digest Collector — نسخه کامل")
    print(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    config_data = load_config()
    provider = get_provider(config_data)

    total_src = len(RSS_SOURCES) + len(NEWSLETTER_SCRAPE) + len(REDDIT_SOURCES) + 7
    print(f"📡 {total_src} منبع فعال")
    print(f"🧠 پرووایدر فعال: {provider.model_id} ({provider.config.get('type')})")
    print("=" * 65)

    raw = collect_all()
    processed = process_items(raw, provider)
    result = save_results(processed, len(raw))

    # محاسبه هزینه واقعی بر اساس قیمت‌های پرووایدر
    cost_in = 0.50
    cost_out = 2.15
    if "gpt-4o-mini" in provider.model_id:
        cost_in = 0.150
        cost_out = 0.600
    elif "gemini" in provider.config.get("type", "") or "ollama" in provider.config.get("type", "") or "localhost" in provider.endpoint_url:
        cost_in = 0.0
        cost_out = 0.0

    cost = result["stats"]["total_processed"] * (450 * cost_in + 200 * cost_out) / 1_000_000
    print(f"""
╔══════════════════════════════════════════╗
║           خلاصه امروز                   ║
╠══════════════════════════════════════════╣
║  مدل: {provider.model_id:<34} ║
║  منابع:        {result['stats']['total_sources']:>26} ║
║  جمع‌آوری:      {result['stats']['total_fetched']:>26} ║
║  پردازش شده:   {result['stats']['total_processed']:>26} ║
║  نمایش داده:   {result['stats']['total_displayed']:>26} ║
║  بالاترین امتیاز: {result['stats']['top_score']:>23.1f} ║
║  وایرال بالا:  {result['stats']['high_viral_count']:>26} ║
║  فرصت عملی:    {result['stats']['opportunities_count']:>26} ║
║  هزینه امروز:  ${cost:>25.4f} ║
╚══════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()

