from .env import *

NUM_WORKERS_PER_QUEUE = 8
CRAWLER_API_PORT = 3002
CRAWLER_SERVICE_NAME = 'crawler-api'
CRAWLER_URL = f'http://{CRAWLER_SERVICE_NAME}:{CRAWLER_API_PORT}'
PLAYWRIGHT_SERVICE_PORT = 3000
PORTS = [PLAYWRIGHT_SERVICE_PORT, CRAWLER_API_PORT]


def compose():
    from .compose import crawler_service, playwright_service, worker
    return {
        "services": {
            CRAWLER_SERVICE_NAME: crawler_service,
            "playwright-service": playwright_service,
            "crawler-worker": worker
        }
    }
