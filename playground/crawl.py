from firecrawl import FirecrawlApp
from infra import crawler

app = FirecrawlApp(api_url=crawler.CRAWLER_URL)

# Scrape a website:
scrape_status = app.scrape_url(
    'https://firecrawl.dev',
    params={'formats': ['markdown', 'html']}
)
print(scrape_status)

# Crawl a website:
crawl_status = app.crawl_url(
    'https://firecrawl.dev',
    params={
        'limit': 100,
        'scrapeOptions': {'formats': ['markdown', 'html']}
    }
)
print(crawl_status)
