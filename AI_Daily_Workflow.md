# 🤖 سیستم پایش روزانه اکوسیستم هوش مصنوعی
### معماری کامل — رایگان، خودکار، با داشبورد وب

---

## 🗺️ نقشه کلی سیستم

```
منابع خام  →  اسکرپر/RSS  →  پردازش هوش مصنوعی  →  ذخیره‌سازی  →  داشبورد وب
   (۷۰+ منبع)    (Python)        (DeepSeek R1/OpenRouter) (JSON file)   (HTML+JS)
```

**کل هزینه ماهانه: $0**  
**زمان اجرای روزانه: ~3-5 دقیقه**  
**نیاز به سرور: GitHub Actions (رایگان) یا Cron محلی**

---

## 📡 فاز ۱: منابع و جمع‌آوری داده

### منابع RSS (بدون نیاز به API)

| منبع | RSS Feed | دسته |
|------|----------|-------|
| MIT Technology Review | `https://www.technologyreview.com/feed/` | تحقیقات |
| VentureBeat AI | `https://venturebeat.com/category/ai/feed/` | صنعت |
| The Verge AI | `https://www.theverge.com/rss/ai-artificial-intelligence/index.xml` | وایرال |
| TechCrunch AI | `https://techcrunch.com/category/artificial-intelligence/feed/` | استارتاپ |
| ArXiv CS.AI | `https://rss.arxiv.org/rss/cs.AI` | تحقیقات |
| ArXiv CS.LG | `https://rss.arxiv.org/rss/cs.LG` | تحقیقات |
| Hugging Face Papers | `https://huggingface.co/blog/feed.xml` | مدل‌ها |
| Google AI Blog | `https://blog.research.google/feeds/posts/default` | مدل‌ها |
| OpenAI Blog | `https://openai.com/blog/rss.xml` | مدل‌ها |
| IEEE Spectrum AI | `https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss` | عمیق |
| Lablab.ai | `https://lablab.ai/blog/rss.xml` | هکاتون |

### منابع نیازمند اسکرپ (بدون API)

- **Reddit**: `r/MachineLearning`, `r/LocalLLaMA`, `r/artificial` — از طریق `reddit.com/r/xxx/.json`
- **Hugging Face trending models**: `https://huggingface.co/models?sort=trending`
- **LMSYS Arena leaderboard**: اسکرپ صفحه عمومی

---

## 🧠 فاز ۲: پردازش با هوش مصنوعی

### مدل انتخابی: DeepSeek R1 0528 (از طریق OpenRouter — هزینه بسیار کم)

پرامپت اصلی برای هر آیتم:

```
تو یه ویراستار تخصصی هوش مصنوعی هستی که برای مخاطبان فارسی‌زبان محتوا تولید می‌کنی.
این خبر/مقاله رو بررسی کن:

عنوان: {title}
متن: {content}

خروجی رو دقیقاً به این فرمت JSON بده:
{
  "importance_score": 1-10,
  "categories": ["model", "tool", "research", "startup", "hackathon", "viral"],
  "headline_fa": "عنوان جذاب فارسی (max 12 کلمه)",
  "summary_fa": "خلاصه 2 جمله‌ای به فارسی",
  "why_it_matters": "یه جمله: چرا این مهمه؟",
  "opportunity": "آیا فرصت درآمدی یا عملی داره؟ (yes/no + توضیح کوتاه)",
  "viral_potential": "high/medium/low",
  "tags": ["llm", "free-api", "hackathon", ...]
}
فقط JSON بده، هیچ توضیح اضافه‌ای نده.
```

---

## 💾 فاز ۳: ساختار ذخیره‌سازی

فایل `daily_digest.json`:

```json
{
  "date": "2026-06-26",
  "generated_at": "08:00 UTC",
  "stats": {
    "total_sources_checked": 11,
    "total_items_fetched": 340,
    "items_after_filter": 28,
    "top_score": 9.2
  },
  "items": [
    {
      "id": "abc123",
      "source": "ArXiv",
      "url": "https://...",
      "importance_score": 9.2,
      "categories": ["model", "research"],
      "headline_fa": "...",
      "summary_fa": "...",
      "why_it_matters": "...",
      "opportunity": "yes — می‌توان API رایگان آن را تست کرد",
      "viral_potential": "high",
      "tags": ["llm", "open-source"],
      "published_at": "2026-06-26T06:30:00Z"
    }
  ]
}
```

---

## 🖥️ فاز ۴: داشبورد وب

فایل `dashboard.html` — یه صفحه استاتیک که:
- فایل JSON را می‌خواند
- فیلتر بر اساس دسته، امتیاز، پتانسیل وایرال
- مرتب‌سازی بر اساس اهمیت
- نمایش badge رنگی برای هر دسته
- نمایش فرصت‌های درآمدی با آیکون ⚡

---

## ⚙️ فاز ۵: زمان‌بندی خودکار

### گزینه A: GitHub Actions (توصیه شده — کاملاً رایگان)

```yaml
# .github/workflows/daily_ai_digest.yml
name: Daily AI Digest
on:
  schedule:
    - cron: '0 5 * * *'  # هر روز ساعت 5 UTC = 8:30 ایران
  workflow_dispatch:

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install feedparser requests beautifulsoup4
      - name: Run collector
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: python collector.py
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v4
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./output
```

### گزینه B: Cron محلی (اگر سرور یا VPS داری)

```bash
# در crontab -e اضافه کن:
0 8 * * * cd /path/to/project && python collector.py >> logs/daily.log 2>&1
```

---

## 📁 ساختار فایل‌های پروژه

```
ai-digest/
├── collector.py          # اسکرپر + پردازش AI
├── dashboard.html        # داشبورد وب
├── config.json           # تنظیمات منابع و فیلترها
├── output/
│   ├── daily_digest.json # داده امروز
│   └── archive/          # آرشیو روزهای قبل
├── .github/
│   └── workflows/
│       └── daily_ai_digest.yml
└── requirements.txt
```

---

## 🚀 راه‌اندازی (قدم به قدم)

### قدم ۱: کلید API از OpenRouter

1. به `openrouter.ai` برو و ثبت‌نام کن
2. یک API Key بساز
3. کلید را در GitHub Secrets با نام `OPENROUTER_API_KEY` ذخیره کن

### قدم ۲: Repository بساز

```bash
git clone https://github.com/your-username/ai-digest
cd ai-digest
pip install -r requirements.txt
```

### قدم ۳: اجرای محلی و تست

می‌توانید اسکریپت را به صورت یک‌باره یا در حالت وب‌سرور برای به‌روزرسانی آسان از داخل داشبورد اجرا کنید:

**روش الف) اجرای عادی (یک‌باره):**
```bash
python collector.py
# فایل output/daily_digest.json ساخته می‌شود و می‌توانید dashboard.html را باز کنید.
```

**روش ب) اجرای وب‌سرور محلی (پیشنهادی جهت استفاده از دکمه به‌روزرسانی فوری):**
```bash
python collector.py --server
# داشبورد در آدرس http://localhost:8000 در دسترس خواهد بود.
# می‌توانید از دکمه «به‌روزرسانی فوری» در بالای هدر برای واکشی آنلاین و زنده اخبار استفاده کنید.
```


### قدم ۴: فعال کردن GitHub Pages

در تنظیمات repo: **Settings > Pages > Source: gh-pages branch**

داشبورد تو در آدرس `https://username.github.io/ai-digest` در دسترسه.

---

## 📊 فیلترهای هوشمند

سیستم آیتم‌ها را با این منطق فیلتر می‌کند:

```python
# آیتم‌هایی که نمایش داده می‌شن:
importance_score >= 6.5  # OR
viral_potential == "high"  # OR  
opportunity == "yes"  # OR
categories contains "hackathon"
```

---

## 🔔 توسعه‌های آینده (اختیاری)

| قابلیت | ابزار | هزینه |
|---------|-------|--------|
| ارسال به تلگرام | Bot API رایگان | $0 |
| خلاصه هفتگی PDF | ReportLab | $0 |
| ردیابی ترندها در طول زمان | SQLite | $0 |
| امتیازدهی با LMSYS leaderboard | اسکرپ | $0 |
| نوتیفیکیشن آیتم‌های امتیاز ۹+ | Telegram | $0 |

---

## ⚠️ محدودیت‌های مهم

- **هزینه OpenRouter**: مدل DeepSeek R1 بسیار ارزان است و با ۳۰۰ درخواست روزانه حدود ۵ سنت هزینه دارد.
- **GitHub Actions رایگان**: 2000 دقیقه/ماه → کافیه (هر اجرا ~3 دقیقه)
- **Reddit JSON**: گاهی rate limit می‌زند → از `time.sleep(2)` بین درخواست‌ها استفاده کن
- **امنیت داده**: هیچ داده حساسی به مدل ارسال نکنید.

