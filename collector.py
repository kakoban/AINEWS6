"""
AI Daily Digest Collector — نسخه کامل با OpenRouter
مدل انتخابی: DeepSeek R1 0528 (استدلال ۱۰/۱۰ — فارسی ۹/۱۰ — $0.50/$2.15 per 1M)
هزینه واقعی: ~$0.05 برای ۳۰۰ آیتم/روز (بسیار پایین‌تر از بودجه $0.50)
"""

import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import feedparser
import requests
from bs4 import BeautifulSoup

# حل مشکل انکودینگ ترمینال در ویندوز
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ─────────────────────────────────────────
# تنظیمات
# ─────────────────────────────────────────

# تنظیمات پیش‌فرض
OUTPUT_DIR = "output"
MAX_ITEMS_TO_PROCESS = 500
MIN_SCORE_TO_SHOW = 6.0
SEEN_IDS_PATH = os.path.join("output", "seen_ids.json")


def load_env():
    env_path = ".env"
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip().strip('"').strip("'")
        except Exception as e:
            print(f"⚠️ خطا در خواندن فایل .env: {e}")


def load_config():
    global OUTPUT_DIR, MAX_ITEMS_TO_PROCESS, MIN_SCORE_TO_SHOW
    load_env()
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                OUTPUT_DIR = cfg.get("output_dir", OUTPUT_DIR)
                MAX_ITEMS_TO_PROCESS = cfg.get(
                    "max_items_to_process", MAX_ITEMS_TO_PROCESS
                )
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

        # هشدار در صورت خالی بودن کلید API برای پرووایدرهای غیر محلی
        is_local = "localhost" in self.endpoint_url or "127.0.0.1" in self.endpoint_url
        if not self.api_key and not is_local:
            print(f"\n⚠️  [هشدار] کلید API برای مدل '{self.model_id}' یافت نشد!")
            print(
                f"   لطفاً متغیر محیطی '{api_key_env}' را تنظیم کنید یا یک فایل '.env' در پوشه اصلی بسازید.\n"
            )

    def analyze(self, item, prompt):
        raise NotImplementedError

    def _parse_json(self, text):
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
            text = text[start : end + 1]
        return json.loads(text)


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
            r = requests.post(
                self.endpoint_url, headers=headers, json=payload, timeout=40
            )
            r.raise_for_status()
            data = r.json()
            text = data["choices"][0]["message"]["content"].strip()
            return self._parse_json(text)
        except Exception as e:
            print(f"    ⚠️ خطا در پرووایدر OpenAI-Compatible: {e}")
            return None


class GeminiProvider(BaseProvider):
    def analyze(self, item, prompt):
        url = f"{self.endpoint_url}/{self.model_id}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": self.temperature,
            },
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


def get_provider(config_data):
    active = config_data.get("active_provider", "openrouter")
    prov_cfg = config_data.get("providers", {}).get(active)
    if not prov_cfg:
        print(
            f"⚠️ پرووایدر '{active}' یافت نشد. از تنظیمات پیش‌فرض OpenRouter استفاده می‌شود."
        )
        prov_cfg = {
            "type": "openai_compatible",
            "endpoint_url": "https://openrouter.ai/api/v1/chat/completions",
            "api_key_env": "OPENROUTER_API_KEY",
            "model_id": "deepseek/deepseek-r1-0528",
            "temperature": 0.3,
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
    {
        "name": "MIT Technology Review",
        "url": "https://www.technologyreview.com/feed/",
        "category": "research",
    },
    {
        "name": "IEEE Spectrum AI",
        "url": "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss",
        "category": "research",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
        "category": "startup",
    },
    {
        "name": "The Verge AI",
        "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        "category": "viral",
    },
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "startup",
    },
    {
        "name": "Wired AI",
        "url": "https://www.wired.com/feed/rss",
        "category": "viral",
    },
    {
        "name": "ZDNet AI",
        "url": "https://www.zdnet.com/topic/artificial-intelligence/rss.xml",
        "category": "research",
    },
    {
        "name": "Ars Technica",
        "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "category": "viral",
    },
    {
        "name": "The Register",
        "url": "https://www.theregister.com/headlines.atom",
        "category": "viral",
    },
    {
        "name": "The Decoder",
        "url": "https://the-decoder.com/feed/",
        "category": "viral",
    },
    {
        "name": "Towards AI",
        "url": "https://pub.towardsai.net/feed",
        "category": "research",
    },
    {
        "name": "Towards Data Science",
        "url": "https://towardsdatascience.com/feed",
        "category": "research",
    },
    # ── وبلاگ‌های رسمی آزمایشگاه‌ها ──
    {
        "name": "Google AI Blog",
        "url": "https://blog.google/technology/ai/rss/",
        "category": "model",
    },
    {
        "name": "Google Research",
        "url": "https://research.google/blog/rss/",
        "category": "model",
    },
    {
        "name": "DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "category": "model",
    },
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "category": "model",
    },
    {
        "name": "Hugging Face Blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "category": "model",
    },
    {
        "name": "Microsoft AI Blog",
        "url": "https://azure.microsoft.com/en-us/blog/feed/",
        "category": "model",
    },
    {
        "name": "NVIDIA AI Blog",
        "url": "https://blogs.nvidia.com/blog/category/deep-learning/feed/",
        "category": "model",
    },
    {
        "name": "AWS ML Blog",
        "url": "https://aws.amazon.com/blogs/machine-learning/feed/",
        "category": "tool",
    },
    {
        "name": "Cloudflare AI Blog",
        "url": "https://blog.cloudflare.com/rss/",
        "category": "tool",
    },
    {
        "name": "DeepSeek Releases",
        "url": "https://github.com/deepseek-ai/DeepSeek-V3/releases.atom",
        "category": "model",
    },
    # ── رسانه‌های عمومی فناوری ──
    {
        "name": "Axios AI",
        "url": "https://api.axios.com/feed/",
        "category": "research",
    },
    {
        "name": "Fast Company",
        "url": "https://www.fastcompany.com/latest/rss",
        "category": "startup",
    },
    {
        "name": "Engadget",
        "url": "https://www.engadget.com/rss.xml",
        "category": "viral",
    },
    {
        "name": "Last Week in AI",
        "url": "https://lastweekin.ai/feed",
        "category": "research",
    },
    # ── Hacker News ──
    {
        "name": "Hacker News Front",
        "url": "https://hnrss.org/frontpage",
        "category": "tool",
    },
    # ── یوتیوب آموزشی ──
    {
        "name": "Two Minute Papers",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCbfYPyITQ-7l4upoX8nvctg",
        "category": "research",
    },
    {
        "name": "AI Explained",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCNJ1Ymd5yFuUPtn21xtRbbw",
        "category": "research",
    },
    {
        "name": "Vercel Blog",
        "url": "https://vercel.com/atom",
        "category": "tool",
    },
    # ── منابع جدید — ژوئیه ۲۰۲۶ ──
    # رسانه‌های تحقیقاتی
    {
        "name": "Nature Machine Intelligence",
        "url": "https://www.nature.com/natmachintell.rss",
        "category": "research",
    },
    {
        "name": "AI2 (Allen Institute)",
        "url": "https://allenai.org/feed.xml",
        "category": "research",
    },
    {
        "name": "Hugging Face Papers",
        "url": "https://huggingface.co/papers/rss",
        "category": "research",
    },
    {
        "name": "ArXiv CS.AI",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "category": "research",
    },
    {
        "name": "ArXiv CS.LG",
        "url": "https://rss.arxiv.org/rss/cs.LG",
        "category": "research",
    },
    {
        "name": "ArXiv CS.CV",
        "url": "https://rss.arxiv.org/rss/cs.CV",
        "category": "research",
    },
    {
        "name": "Papers With Code",
        "url": "https://paperswithcode.com/rss",
        "category": "research",
    },
    # وبلاگ‌های آزمایشگاه‌ها
    {
        "name": "Meta AI Blog",
        "url": "https://ai.meta.com/blog/rss/",
        "category": "model",
    },
    {
        "name": "Anthropic Blog",
        "url": "https://www.anthropic.com/rss.xml",
        "category": "model",
    },
    {
        "name": "Mistral AI Blog",
        "url": "https://mistral.ai/feed.xml",
        "category": "model",
    },
    {
        "name": "Cohere Blog",
        "url": "https://txt.cohere.com/rss/",
        "category": "model",
    },
    {
        "name": "Stability AI Blog",
        "url": "https://stability.ai/blog/rss.xml",
        "category": "model",
    },
    # ابزارها و فریم‌ورک‌ها
    {
        "name": "Weights & Biases Blog",
        "url": "https://wandb.ai/fully-connected/feed",
        "category": "tool",
    },
    {
        "name": "LangChain Blog",
        "url": "https://blog.langchain.dev/rss/",
        "category": "tool",
    },
    {
        "name": "LlamaIndex Blog",
        "url": "https://www.llamaindex.ai/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Pinecone Blog",
        "url": "https://www.pinecone.io/blog/rss/",
        "category": "tool",
    },
    {
        "name": "Replicate Blog",
        "url": "https://replicate.com/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Groq Blog",
        "url": "https://groq.com/blog/feed/",
        "category": "tool",
    },
    {
        "name": "Together AI Blog",
        "url": "https://www.together.ai/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Ollama Blog",
        "url": "https://ollama.com/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Cursor Blog",
        "url": "https://cursor.sh/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Replit Blog",
        "url": "https://blog.replit.com/blog/rss.xml",
        "category": "tool",
    },
    # وکتور دیتابیس‌ها
    {
        "name": "Qdrant Blog",
        "url": "https://qdrant.tech/blog/rss/",
        "category": "tool",
    },
    {
        "name": "Weaviate Blog",
        "url": "https://weaviate.io/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "ChromaDB Blog",
        "url": "https://blog.trychroma.com/rss",
        "category": "tool",
    },
    # MLOps و دیتا
    {
        "name": "Databricks Blog",
        "url": "https://www.databricks.com/blog/rss",
        "category": "tool",
    },
    {
        "name": "MLflow Blog",
        "url": "https://mlflow.org/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Roboflow Blog",
        "url": "https://blog.roboflow.com/rss/",
        "category": "tool",
    },
    {
        "name": "PyTorch Blog",
        "url": "https://pytorch.org/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "TensorFlow Blog",
        "url": "https://blog.tensorflow.org/rss.xml",
        "category": "tool",
    },
    # یوتیوب آموزشی بیشتر
    {
        "name": "Yannic Kilcher",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCZHmQk67mSJgfCCTn7xBfew",
        "category": "research",
    },
    {
        "name": "AI Jason",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCV206gMIkUJ2mJiGk8-YOeg",
        "category": "tool",
    },
    {
        "name": "Matthew Berman",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCX6OQ3DkcsbYNE6H8uQQuVA",
        "category": "viral",
    },
    {
        "name": "Fireship",
        "url": "https://www.youtube.com/feeds/videos.xml?channel_id=UCsBjURrPoezykLs9EqgamOA",
        "category": "tool",
    },
    # رسانه‌های فارسی
    {
        "name": "Zoomit",
        "url": "https://www.zoomit.ir/feed/",
        "category": "viral",
    },
    {
        "name": "Digiatoo",
        "url": "https://digiato.com/feed",
        "category": "viral",
    },
    {
        "name": "Zoomg",
        "url": "https://www.zoomg.ir/feed",
        "category": "viral",
    },
    # ابزارهای بیشتر
    {
        "name": "RunPod Blog",
        "url": "https://www.runpod.io/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Modal Blog",
        "url": "https://modal.com/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Supabase Blog",
        "url": "https://supabase.com/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Neon Blog",
        "url": "https://neon.tech/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Prisma Blog",
        "url": "https://www.prisma.io/blog/rss.xml",
        "category": "tool",
    },
    {
        "name": "Redis Blog",
        "url": "https://redis.io/blog/rss/",
        "category": "tool",
    },
    {
        "name": "Elastic Blog",
        "url": "https://www.elastic.co/blog/feed",
        "category": "tool",
    },
    {
        "name": "Snowflake Blog",
        "url": "https://www.snowflake.com/blog/rss/",
        "category": "tool",
    },
    {
        "name": "OpenCV Blog",
        "url": "https://opencv.org/blog/rss/",
        "category": "tool",
    },

    # ── arXiv — تحقیقات علمی (فقط حوزه‌های کاربردی) ──
    {
        "name": "ArXiv CS.CL",
        "url": "https://rss.arxiv.org/rss/cs.CL",
        "category": "research",
    },
    # ── اکانت‌های برتر X/Twitter — اخبار مدل‌های رایگان (از طریق Nitter RSS) ──
    {
        "name": "X/@GoogleDeepMind",
        "url": "https://nitter.net/GoogleDeepMind/rss",
        "category": "model",
    },
    {
        "name": "X/@OpenAI",
        "url": "https://nitter.net/OpenAI/rss",
        "category": "model",
    },
    {
        "name": "X/@_akhaliq",
        "url": "https://nitter.net/_akhaliq/rss",
        "category": "model",
    },
    {
        "name": "X/@NousResearch",
        "url": "https://nitter.net/NousResearch/rss",
        "category": "model",
    },
    {
        "name": "X/@huggingface",
        "url": "https://nitter.net/huggingface/rss",
        "category": "model",
    },
    {
        "name": "X/@ollama",
        "url": "https://nitter.net/ollama/rss",
        "category": "tool",
    },
    {
        "name": "X/@AnthropicAI",
        "url": "https://nitter.net/AnthropicAI/rss",
        "category": "model",
    },
    {
        "name": "X/@MistralAI",
        "url": "https://nitter.net/MistralAI/rss",
        "category": "model",
    },
    {
        "name": "X/@svpino",
        "url": "https://nitter.net/svpino/rss",
        "category": "tool",
    },
    {
        "name": "X/@togethercompute",
        "url": "https://nitter.net/togethercompute/rss",
        "category": "free-resource",
    },
    # ── اکانت‌های فارسی‌زبان ──
    {
        "name": "X/@naviidtaheri",
        "url": "https://nitter.net/naviidtaheri/rss",
        "category": "tool",
    },
    {
        "name": "X/@bookunt",
        "url": "https://nitter.net/bookunt/rss",
        "category": "tool",
    },
    {
        "name": "X/@amirhosseinssl",
        "url": "https://nitter.net/amirhosseinssl/rss",
        "category": "research",
    },
]

# ─────────────────────────────────────────
# ۳. Reddit — RSS فید عمومی (بدون نیاز به API key)
# ─────────────────────────────────────────
REDDIT_SOURCES = [
    # آکادمیک و تحقیقاتی
    {"subreddit": "MachineLearning", "category": "research"},
    {"subreddit": "LocalLLaMA", "category": "model"},
    {"subreddit": "artificial", "category": "viral"},
    {"subreddit": "singularity", "category": "viral"},
    # کاربران عمومی و وایرال
    {"subreddit": "ChatGPT", "category": "viral"},
    {"subreddit": "ChatGPTPro", "category": "tool"},
    # توسعه‌دهندگان
    {"subreddit": "VibeCoding", "category": "tool"},
    {"subreddit": "ClaudeAI", "category": "model"},
    {"subreddit": "ChatGPTCoding", "category": "tool"},
]

# ─────────────────────────────────────────
# ۴. منابع تخصصی — API / اسکرپ مستقیم
# ─────────────────────────────────────────
SPECIAL_SOURCES = [
    # مدل‌های ترند
    {
        "name": "HF Trending Models",
        "type": "hf_trending",
        "url": "https://huggingface.co/api/models?sort=trendingScore&limit=10",
        "category": "model",
    },
    # GitHub — سیگنال اولیه ترندها
    {
        "name": "GitHub Trending AI",
        "type": "github_trending",
        "url": "https://github.com/trending?l=python&since=daily",
        "category": "tool",
        "note": "ریپوهای ترند روزانه پایتون — اولین سیگنال ابزارهای جدید",
    },
    # GitHub Awesome Lists — کیوریت‌شده
    {
        "name": "awesome-evals (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/benchflow-ai/awesome-evals/commits",
        "category": "research",
        "note": "۴۴۳+ منبع ارزیابی عامل‌های AI — به‌روزرسانی چند بار در هفته",
    },
    {
        "name": "awesome-opensource-ai (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/alvinreal/awesome-opensource-ai/commits",
        "category": "tool",
        "note": "پروژه‌های متن‌باز AI — به‌روزرسانی روزانه",
    },
    {
        "name": "awesome-ai-agents-2026 (GitHub commits)",
        "type": "github_commits",
        "url": "https://api.github.com/repos/caramaschiHG/awesome-ai-agents-2026/commits",
        "category": "tool",
        "note": "۳۰۰+ فریم‌ورک و ابزار Agentic AI",
    },
    # منابع رایگان API
    {
        "name": "OpenRouter Free Models",
        "type": "openrouter_free",
        "url": "https://openrouter.ai/api/v1/models",
        "category": "free-resource",
        "note": "لیست مدل‌های رایگان OpenRouter — آپدیت روزانه",
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
        for entry in feed.entries[:20]:
            summary = entry.get("summary", entry.get("description", ""))
            clean_text = BeautifulSoup(summary, "html.parser").get_text(
                separator=" ", strip=True
            )[:800]
            items.append(
                {
                    "id": hashlib.md5(
                        (entry.get("link", "") + entry.get("title", "")).encode()
                    ).hexdigest()[:10],
                    "source": source["name"],
                    "base_category": source["category"],
                    "title": entry.get("title", "")[:200],
                    "url": entry.get("link", ""),
                    "summary_raw": clean_text,
                    "published_at": entry.get(
                        "published", datetime.now(timezone.utc).isoformat()
                    ),
                }
            )
        print(f"  ✅ {source['name']}: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ {source['name']}: {e}")
        return []




def fetch_reddit(config):
    """خواندن Reddit از طریق RSS فید عمومی"""
    url = f"https://www.reddit.com/r/{config['subreddit']}/hot.rss"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        feed = feedparser.parse(resp.content)
        items = []
        for entry in feed.entries[:20]:
            title = entry.get("title", "")
            link = entry.get("link", "")
            summary = BeautifulSoup(
                entry.get("summary", entry.get("description", "")), "html.parser"
            ).get_text(separator=" ", strip=True)[:600]
            items.append(
                {
                    "id": hashlib.md5(link.encode()).hexdigest()[:10],
                    "source": f"r/{config['subreddit']}",
                    "base_category": config.get("category", "viral"),
                    "title": title[:200],
                    "url": link,
                    "summary_raw": summary,
                    "published_at": entry.get(
                        "published", datetime.now(timezone.utc).isoformat()
                    ),
                }
            )
        time.sleep(1.5)
        print(f"  ✅ r/{config['subreddit']}: {len(items)} آیتم")
        return items
    except Exception as e:
        print(f"  ❌ r/{config['subreddit']}: {e}")
        return []


def fetch_hf_trending():
    """مدل‌های ترند Hugging Face"""
    try:
        r = requests.get(
            "https://huggingface.co/api/models?sort=trendingScore&limit=10",
            timeout=10,
        )
        models = r.json()
        if not isinstance(models, list):
            print(f"  ❌ HF Trending: Expected list response, got {type(models)}")
            return []
        items = []
        for m in models:
            name = m.get("modelId", m.get("id", ""))
            items.append(
                {
                    "id": hashlib.md5(name.encode()).hexdigest()[:10],
                    "source": "HF Trending Models",
                    "base_category": "model",
                    "title": f"مدل ترند: {name}",
                    "url": f"https://huggingface.co/{name}",
                    "summary_raw": f"likes: {m.get('likes', 0)} | downloads: {m.get('downloads', 0)} | pipeline: {m.get('pipeline_tag', '')}",
                    "published_at": m.get(
                        "lastModified", datetime.now(timezone.utc).isoformat()
                    ),
                }
            )
        print(f"  ✅ HF Trending: {len(items)} مدل")
        return items
    except Exception as e:
        print(f"  ❌ HF Trending: {e}")
        return []



def fetch_github_trending():
    """ریپوهای ترند روزانه Python در GitHub — سیگنال اولیه ابزارهای جدید"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AI-Digest-Bot/1.0)"}
    try:
        r = requests.get(
            "https://github.com/trending/python?since=daily",
            headers=headers,
            timeout=12,
        )
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
            ai_keywords = [
                "llm",
                "ai",
                "gpt",
                "model",
                "agent",
                "ml",
                "neural",
                "transformer",
                "diffusion",
                "rag",
                "embed",
                "inference",
                "openai",
                "claude",
                "gemini",
                "llama",
                "deepseek",
            ]
            combined = (name + " " + desc).lower()
            if not any(kw in combined for kw in ai_keywords):
                continue

            items.append(
                {
                    "id": hashlib.md5(url.encode()).hexdigest()[:10],
                    "source": "GitHub Trending AI",
                    "base_category": "tool",
                    "title": f"GitHub Trending: {name} — ⭐{stars}",
                    "url": url,
                    "summary_raw": desc,
                    "published_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        print(f"  ✅ GitHub Trending AI: {len(items)} ریپو")
        return items
    except Exception as e:
        print(f"  ❌ GitHub Trending: {e}")
        return []


def fetch_github_awesome_commits(cfg):
    """آخرین تغییرات در لیست‌های Awesome GitHub"""
    try:
        headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "AI-Digest-Bot/1.0",
        }
        gh_token = os.environ.get("GITHUB_TOKEN", "")
        if gh_token:
            headers["Authorization"] = f"Bearer {gh_token}"
        r = requests.get(cfg["url"] + "?per_page=5", headers=headers, timeout=10)
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
            items.append(
                {
                    "id": hashlib.md5(sha.encode()).hexdigest()[:10],
                    "source": cfg["name"],
                    "base_category": cfg["category"],
                    "title": f"[{repo_name.split('/')[-1]}] {msg}",
                    "url": url,
                    "summary_raw": cfg.get("note", "") + " — " + msg,
                    "published_at": commit.get("commit", {})
                    .get("committer", {})
                    .get("date", datetime.now(timezone.utc).isoformat()),
                }
            )
        print(f"  ✅ {cfg['name']}: {len(items)} کامیت جدید")
        return items
    except Exception as e:
        print(f"  ❌ {cfg['name']}: {e}")
        return []


def fetch_openrouter_free_models():
    """لیست مدل‌های رایگان OpenRouter از API رسمی"""
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"User-Agent": "AI-Digest-Bot/1.0"},
            timeout=10,
        )
        data = r.json()
        models = data.get("data", [])
        free_models = [m for m in models if m.get("pricing", {}).get("prompt") == "0"]
        if not free_models:
            return []
        names = ", ".join([m.get("name", m.get("id", "")) for m in free_models[:10]])
        items = [
            {
                "id": hashlib.md5("openrouter_free_today".encode()).hexdigest()[:10],
                "source": "OpenRouter Free Models",
                "base_category": "free-resource",
                "title": f"مدل‌های رایگان OpenRouter امروز: {len(free_models)} مدل",
                "url": "https://openrouter.ai/models?q=free",
                "summary_raw": f"مدل‌های رایگان: {names}",
                "published_at": datetime.now(timezone.utc).isoformat(),
            }
        ]
        print(f"  ✅ OpenRouter Free: {len(free_models)} مدل رایگان")
        return items
    except Exception as e:
        print(f"  ❌ OpenRouter Free: {e}")
        return []


# ─────────────────────────────────────────
# جمع‌آوری کل
# ─────────────────────────────────────────


def load_seen_ids():
    """بارگذاری شناسه‌های آیتم‌های دیده‌شده برای جلوگیری از پردازش تکراری"""
    if os.path.exists(SEEN_IDS_PATH):
        try:
            with open(SEEN_IDS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                # داده قدیمی‌تر از ۱۴ روز حذف می‌شود
                cutoff = datetime.now(timezone.utc).timestamp() - 14 * 86400
                fresh = {k: v for k, v in data.items() if v >= cutoff}
                return fresh
        except Exception:
            pass
    return {}


def save_seen_ids(seen_dict, new_ids):
    """ذخیره شناسه‌های جدید در کش"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    now_ts = datetime.now(timezone.utc).timestamp()
    merged = {**seen_dict, **{i: now_ts for i in new_ids}}
    try:
        with open(SEEN_IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(merged, f)
    except Exception as e:
        print(f"  ⚠️ خطا در ذخیره seen_ids: {e}")


def collect_all():
    print("\n📡 مرحله ۱: جمع‌آوری موازی داده‌ها\n")
    all_items = []

    print("── RSS رسانه‌های رسمی و آزمایشگاه‌ها (موازی) ──")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_rss, src): src for src in RSS_SOURCES}
        for future in as_completed(futures):
            try:
                all_items.extend(future.result())
            except Exception as e:
                print(f"  ⚠️ خطا در RSS: {e}")

    print("\n── Reddit ──")
    for cfg in REDDIT_SOURCES:
        all_items.extend(fetch_reddit(cfg))

    print("\n── منابع تخصصی ──")
    all_items.extend(fetch_hf_trending())

    print("\n── GitHub (ترند + Awesome Lists) ──")
    all_items.extend(fetch_github_trending())
    for src in SPECIAL_SOURCES:
        if src.get("type") == "github_commits":
            all_items.extend(fetch_github_awesome_commits(src))
            time.sleep(1)

    print("\n── OpenRouter Free Models ──")
    all_items.extend(fetch_openrouter_free_models())

    # حذف تکراری بر اساس ID و عنوان مشابه
    seen = set()
    seen_titles = set()
    unique = []
    for item in all_items:
        if item["id"] in seen:
            continue
        # حذف تکراری بر اساس عنوان مشابه (۴۰ کاراکتر اول)
        title_key = item.get("title", "")[:40].lower().strip()
        if title_key in seen_titles:
            continue
        seen.add(item["id"])
        seen_titles.add(title_key)
        unique.append(item)

    # فیلتر کیفیت: حذف RT، Reply، و محتوای قدیمی
    filtered = []
    cutoff = datetime.now(timezone.utc).timestamp() - 48 * 3600  # ۴۸ ساعت اخیر
    rt_skipped = 0
    old_skipped = 0
    low_skipped = 0

    for item in unique:
        title = item.get("title", "")

        # حذف Retweet و Reply
        if title.startswith("RT by ") or title.startswith("R to @"):
            rt_skipped += 1
            continue

        # حذف توییت‌های خیلی کوتاه (کمتر از ۱۵ کاراکتر)
        if len(title.strip()) < 15:
            low_skipped += 1
            continue

        # حذف آیتم‌های قدیمی‌تر از ۴۸ ساعت
        pub = item.get("published_at", "")
        try:
            pub_dt = None
            # ISO format: 2026-07-04T08:20:09.362389+00:00
            if 'T' in str(pub):
                pub_dt = datetime.fromisoformat(str(pub).replace('Z', '+00:00'))
            else:
                # RFC 2822 format: Fri, 03 Jul 2026 16:16:27 GMT
                from email.utils import parsedate_to_datetime
                pub_dt = parsedate_to_datetime(pub)
            if pub_dt and pub_dt.timestamp() < cutoff:
                old_skipped += 1
                continue
        except Exception:
            # اگر تاریخ قابل تشخیص نبود، حذفش کن (نه نگهش دار)
            old_skipped += 1
            continue

        filtered.append(item)

    if rt_skipped or old_skipped or low_skipped:
        print(f"\n🔽 فیلتر کیفیت: حذف {rt_skipped} RT/Reply، {old_skipped} قدیمی، {low_skipped} کوتاه")

    total_sources = (
        len(RSS_SOURCES)
        + len(REDDIT_SOURCES)
        + len(SPECIAL_SOURCES)
    )
    print(
        f"\n📊 {total_sources} منبع فعال — {len(all_items)} آیتم → {len(unique)} یکتا → {len(filtered)} باکیفیت"
    )
    return filtered


# ─────────────────────────────────────────
# پردازش Gemini
# ─────────────────────────────────────────

PRIORITY_KEYWORDS = [
    "release",
    "launch",
    "open-source",
    "free",
    "benchmark",
    "hackathon",
    "llm",
    "gpt",
    "claude",
    "gemini",
    "llama",
    "deepseek",
    "qwen",
    "agent",
    "model",
    "api",
    "paper",
    "funding",
    "billion",
    "state-of-the-art",
    "trending",
    "sota",
    "fine-tun",
    "quantiz",
    "rlhf",
    "multimodal",
    "reasoning",
    "agentic",
    "autonomous",
    "breakthrough",
    "leaderboard",
]


def priority_score(item):
    score = sum(1 for kw in PRIORITY_KEYWORDS if kw in item["title"].lower())
    if item.get("score_reddit", 0) > 500:
        score += 3
    if item["base_category"] in ["model", "hackathon"]:
        score += 2
    if item["base_category"] == "research":
        score += 1
    return score


def analyze_item(item, provider):
    """تحلیل هر آیتم با پرووایدر هوش مصنوعی فعال"""
    prompt = f"""تو یه ویراستار تخصصی هوش مصنوعی هستی که برای توسعه‌دهندگان فارسی‌زبان خبر جمع‌آوری می‌کنی.
این خبر/پست رو بررسی کن:

منبع: {item["source"]}
عنوان: {item["title"]}
محتوا: {item.get("summary_raw", "")[:500]}

⚠️ قوانین مهم فیلتر:
- اگه خبر اصلاً ربطی به هوش مصنوعی، یادگیری ماشین، LLM، رباتیک، یا تکنولوژی نداره → امتیاز ۱ بده
- اگه محتوا صرفاً تبلیغات، سرگرمی محض، یا clickbait بدون ارزش فنی هست → امتیاز ۲ بده
- عنوان خبر باید دقیقاً مربوط به محتوای AI باشه، نه صرفاً از منبع AI

قوانین امتیازدهی:
- امتیاز ۹-۱۰: انتشار مدل جدید رایگان/متن‌باز، API رایگان جدید، هکاتون با جایزه نقدی
- امتیاز ۷-۸: آپدیت مهم مدل‌ها، ابزار جدید کاربردی، خبر مهم صنعت AI
- امتیاز ۵-۶: مقاله جالب AI، آموزش ML، تحلیل فنی
- امتیاز ۱-۴: محتوای عمومی، تکراری، غیرمرتبط با AI، بی‌فایده

اگه خبر درباره مدل رایگان یا ابزار رایگان هست، حتماً category شامل "free-resource" باشه و opportunity "yes" باشه.

خروجی رو دقیقاً به این فرمت JSON بده — هیچ توضیح اضافه‌ای نده:
{{
  "importance_score": <عدد 1 تا 10>,
  "categories": <آرایه از: "model", "tool", "research", "startup", "hackathon", "viral", "free-resource">,
  "headline_fa": "<عنوان جذاب فارسی حداکثر ۱۲ کلمه>",
  "summary_fa": "<خلاصه ۲ جمله فارسی: دقیقاً چیه و چطور استفاده کنیم — نه فقط 'معرفی شد'>",
  "why_it_matters": "<یه جمله فارسی: چرا مهمه؟>",
  "opportunity": "<yes یا no — اگر yes: توضیح کوتاه فرصت عملی/درآمدی>",
  "viral_potential": "<high یا medium یا low>",
  "tags": <آرایه تگ‌های انگلیسی کوتاه>
}}"""
    for attempt in range(2):
        result = provider.analyze(item, prompt)
        if result:
            return result
        if attempt == 0:
            time.sleep(2)
    return None



# کلمات کلیدی مرتبط با AI
AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "llm", "large language", "gpt", "claude", "gemini", "llama",
    "deepseek", "qwen", "mistral", "openai", "anthropic", "deepmind",
    "transformer", "neural", "model", "fine-tun", "quantiz", "rlhf",
    "rag", "agent", "agentic", "embedding", "vector", "tokenizer",
    "diffusion", "generative", "multimodal", "reasoning", "benchmark",
    "open-source", "hugging", "gguf", "ggml", "inference", "training",
    "dataset", "paper", "arxiv", "robot", "autonomous", "chatbot",
    "copilot", "cursor", "vibe cod", "mcp", "tool", "api",
    "free model", "trending", "release", "funding", "startup",
]

def has_ai_signal(item):
    text = (item.get("title", "") + " " + item.get("summary_raw", "")[:300]).lower()
    return any(kw in text for kw in AI_KEYWORDS)

def process_items(items, provider):
    print(f"\n🧠 مرحله ۲: پردازش با {provider.model_id}\n")

    # پیش‌فیلتر: حذف آیتم‌های غیرمرتبط با AI
    ai_items = [i for i in items if has_ai_signal(i)]
    skipped_ai = len(items) - len(ai_items)
    if skipped_ai > 0:
        print(f"  ⏭️  {skipped_ai} آیتم غیرمرتبط با AI حذف شد")
    print(f"  📋 {len(ai_items)} آیتم مرتبط با AI")

    sorted_items = sorted(ai_items, key=priority_score, reverse=True)
    to_process = sorted_items[:MAX_ITEMS_TO_PROCESS]

    # تخمین هزینه بر اساس مدل فعال
    cost_in = 0.50
    cost_out = 2.15
    if "gpt-4o-mini" in provider.model_id:
        cost_in = 0.150
        cost_out = 0.600
    elif (
        "gemini" in provider.config.get("type", "")
        or "ollama" in provider.config.get("type", "")
        or "localhost" in provider.endpoint_url
    ):
        cost_in = 0.0
        cost_out = 0.0

    cost_est = len(to_process) * (450 * cost_in + 200 * cost_out) / 1_000_000
    print(f"  پردازش {len(to_process)} آیتم از {len(items)} کل")
    print(f"  تخمین هزینه: ${cost_est:.4f}\n")

    results = []
    for i, item in enumerate(to_process):
        print(f"  [{i + 1}/{len(to_process)}] {item['title'][:55]}...")
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
        item
        for item in items
        if (
            item.get("importance_score", 0) >= MIN_SCORE_TO_SHOW
            or item.get("viral_potential") == "high"
            or str(item.get("opportunity", "no")).startswith("yes")
        )
    ]
    filtered.sort(key=lambda x: x.get("importance_score", 0), reverse=True)

    output = {
        "date": today,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "total_sources": len(RSS_SOURCES)
            + len(REDDIT_SOURCES)
            + len(SPECIAL_SOURCES),
            "total_fetched": total_fetched,
            "total_processed": len(items),
            "total_displayed": len(filtered),
            "top_score": filtered[0]["importance_score"] if filtered else 0,
            "high_viral_count": sum(
                1 for x in filtered if x.get("viral_potential") == "high"
            ),
            "opportunities_count": sum(
                1 for x in filtered if str(x.get("opportunity", "no")).startswith("yes")
            ),
        },
        "items": filtered,
    }

    for path in [
        f"{OUTPUT_DIR}/daily_digest.json",
        f"{OUTPUT_DIR}/archive/{today}.json",
    ]:
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

    # به‌روزرسانی فایل ایندکس آرشیو برای داشبورد
    try:
        archive_dir = f"{OUTPUT_DIR}/archive"
        dates = sorted(
            [
                f[:-5]
                for f in os.listdir(archive_dir)
                if f.endswith(".json") and f != "index.json"
            ],
            reverse=True,
        )
        with open(f"{archive_dir}/index.json", "w", encoding="utf-8") as f:
            json.dump({"dates": dates}, f, ensure_ascii=False)
    except Exception as e:
        print(f"  ⚠️ خطا در ساخت ایندکس آرشیو: {e}")

    return output


def send_telegram_alert(items):
    """ارسال نوتیفیکیشن تلگرام برای آیتم‌های با امتیاز بالا"""
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return

    try:
        min_score = float(os.environ.get("TELEGRAM_MIN_SCORE", "9.0"))
    except ValueError:
        min_score = 9.0

    high_score = [i for i in items if i.get("importance_score", 0) >= min_score]
    if not high_score:
        return

    print(f"\n📬 ارسال {len(high_score)} نوتیفیکیشن تلگرام...")
    for item in high_score[:5]:
        score = item.get("importance_score", 0)
        headline = item.get("headline_fa", item.get("title", ""))
        summary = item.get("summary_fa", "")
        why = item.get("why_it_matters", "")
        url = item.get("url", "")
        opp = item.get("opportunity", "no")
        opp_line = (
            f"\n⚡ {opp.replace('yes — ', '').replace('yes - ', '')}"
            if str(opp).startswith("yes")
            else ""
        )

        text = (
            f"🔴 *امتیاز {score:.1f}/۱۰* — {item.get('source', '')}\n\n"
            f"*{headline}*\n\n"
            f"{summary}\n\n"
            f"💡 {why}"
            f"{opp_line}\n\n"
            f"🔗 {url}"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={
                    "chat_id": chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": False,
                },
                timeout=10,
            )
            time.sleep(0.5)
        except Exception as e:
            print(f"  ⚠️ خطا در ارسال تلگرام: {e}")


# وضعیت جهانی اجرای وظیفه واکشی
_fetch_job = {"status": "idle", "message": "", "total_displayed": 0}
_fetch_lock = threading.Lock()


def _run_fetch_job():
    """اجرای جمع‌آوری اخبار در یک Thread جداگانه"""
    global _fetch_job
    try:
        config_data = load_config()
        provider = get_provider(config_data)
        seen_ids = load_seen_ids()
        raw = collect_all()
        
        # فیلتر آیتم‌های تکراری
        new_items = [i for i in raw if i["id"] not in seen_ids]
        skipped = len(raw) - len(new_items)
        if skipped > 0:
            print(f"⏭️  {skipped} آیتم قبلاً پردازش شده (حذف از صف AI)")

        processed = process_items(new_items, provider)
        result = save_results(processed, len(raw))

        # ذخیره کش شناسه‌ها — فقط آیتم‌هایی که امتیاز بالا دارن
        high_score_ids = {i["id"] for i in processed if i.get("importance_score", 0) >= MIN_SCORE_TO_SHOW}
        save_seen_ids(seen_ids, high_score_ids)
        # ارسال نوتیفیکیشن تلگرام
        send_telegram_alert(result["items"])

        displayed = len(result["items"])
        with _fetch_lock:
            _fetch_job = {
                "status": "success",
                "message": f"{len(processed)} خبر جدید پردازش و به‌روزرسانی شد.",
                "total_displayed": displayed,
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
                res = {
                    "status": "running",
                    "message": "واکشی اخبار از قبل در حال اجراست. لطفاً صبر کنید.",
                }
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
                return

            # ریست وضعیت و شروع thread جدید
            with _fetch_lock:
                _fetch_job = {
                    "status": "running",
                    "message": "در حال جمع‌آوری اخبار...",
                    "total_displayed": 0,
                }

            t = threading.Thread(target=_run_fetch_job, daemon=True)
            t.start()
            print("\n🔄 thread واکشی اخبار شروع شد...")

            # پاسخ فوری به مرورگر
            self.send_response(202)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            res = {
                "status": "running",
                "message": "واکشی اخبار شروع شد. وضعیت را از /api/status دنبال کنید.",
            }
            self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
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
            self.wfile.write(json.dumps(state, ensure_ascii=False).encode("utf-8"))
            return

        # هندل کردن درخواست favicon برای جلوگیری از خطای 404
        if clean_path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
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
                self.wfile.write(json.dumps(res, ensure_ascii=False).encode("utf-8"))
            return

        # سایر فایل‌های استاتیک
        super().do_GET()


def start_server(port=8000):
    server_address = ("", port)
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
    seen_ids = load_seen_ids()

    total_src = (
        len(RSS_SOURCES)
        + len(REDDIT_SOURCES)
        + len(SPECIAL_SOURCES)
    )
    print(f"📡 {total_src} منبع فعال")
    print(f"🧠 پرووایدر فعال: {provider.model_id} ({provider.config.get('type')})")
    print("=" * 65)

    raw = collect_all()

    # فیلتر آیتم‌های تکراری از روزهای قبل
    new_items = [i for i in raw if i["id"] not in seen_ids]
    skipped = len(raw) - len(new_items)
    if skipped > 0:
        print(f"⏭️  {skipped} آیتم قبلاً پردازش شده (حذف از صف AI)")

    processed = process_items(new_items, provider)
    result = save_results(processed, len(raw))

    # ذخیره کش شناسه‌ها
    save_seen_ids(seen_ids, {i["id"] for i in raw})
    # ارسال نوتیفیکیشن تلگرام
    send_telegram_alert(result["items"])

    # محاسبه هزینه واقعی بر اساس قیمت‌های پرووایدر
    cost_in = 0.50
    cost_out = 2.15
    if "gpt-4o-mini" in provider.model_id:
        cost_in = 0.150
        cost_out = 0.600
    elif (
        "gemini" in provider.config.get("type", "")
        or "ollama" in provider.config.get("type", "")
        or "localhost" in provider.endpoint_url
    ):
        cost_in = 0.0
        cost_out = 0.0

    cost = (
        result["stats"]["total_processed"]
        * (450 * cost_in + 200 * cost_out)
        / 1_000_000
    )
    print(f"""
╔══════════════════════════════════════════╗
║           خلاصه امروز                   ║
╠══════════════════════════════════════════╣
║  مدل: {provider.model_id:<34} ║
║  منابع:        {result["stats"]["total_sources"]:>26} ║
║  جمع‌آوری:      {result["stats"]["total_fetched"]:>26} ║
║  پردازش شده:   {result["stats"]["total_processed"]:>26} ║
║  نمایش داده:   {result["stats"]["total_displayed"]:>26} ║
║  بالاترین امتیاز: {result["stats"]["top_score"]:>23.1f} ║
║  وایرال بالا:  {result["stats"]["high_viral_count"]:>26} ║
║  فرصت عملی:    {result["stats"]["opportunities_count"]:>26} ║
║  هزینه امروز:  ${cost:>25.4f} ║
╚══════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
