import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")


def get_company_news(query="GOOG", limit=5):
    """
    Fetches latest news articles using a specific query string (symbol or full company name).
    Always returns a list. Never raises exceptions.
    """

    # ✅ Safety: Key missing
    if not NEWS_API_KEY:
        print("Warning: NEWS_API_KEY is not set. News fetching skipped.")
        return []

    # ✅ Quotes improve multi-word search
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q=\"{query}\"&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url)
        # ✅ If API returns HTML (rate limit, outage)
        if response.headers.get("content-type", "").startswith("text/html"):
            print("NewsAPI returned HTML instead of JSON → ignoring.")
            return []

        resp = response.json()

    except Exception as e:
        print(f"Exception while fetching news: {e}")
        return []

    # ✅ Safety: API error or invalid key
    if resp.get("status") != "ok":
        print(f"NewsAPI Error: {resp.get('message')}")
        return []

    # ✅ Extract clean articles
    articles = []
    for a in resp.get("articles", []):
        if len(articles) >= limit:
            break

        title = a.get("title")
        desc = a.get("description")

        # ✅ Safety: ensure both title & description exist AND are strings
        if not title or not isinstance(title, str):
            continue
        if not desc or not isinstance(desc, str):
            continue

        articles.append({
            "title": title.strip(),
            "description": desc.strip(),
            "url": a.get("url", "")
        })

    print(f"Fetched {len(articles)} articles from NewsAPI for query: \"{query}\"")
    return articles
