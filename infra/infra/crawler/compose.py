import yaml
from util import workspace
from infra.db import kvs
from . import *

yaml_data = workspace.read("apps/firecrawl/docker-compose.yaml")
compose_data = yaml.safe_load(yaml_data)

crawler_service = compose_data["services"]["api"]
worker = compose_data["services"]["worker"]

for item in [crawler_service, worker]:
    item["build"] = workspace.relpath(
        __file__, "apps/firecrawl/" + item["build"])
    item.pop("networks")
    item.pop("extra_hosts")
    item["environment"]["REDIS_URL"] = kvs.REDIS_URL
    item["environment"]["REDIS_RATE_LIMIT_URL"] = kvs.REDIS_URL
    item["environment"]["USE_DB_AUTHENTICATION"] = False
    item["environment"]["PLAYWRIGHT_MICROSERVICE_URL"] = "http://playwright-service:3000/html"
    item["environment"]["PORT"] = CRAWLER_API_PORT

worker["depends_on"] = ["crawler-api", "playwright-service", "redis"]

playwright_service = compose_data["services"]["playwright-service"]
playwright_service.pop("environment")
playwright_service.pop("networks")
playwright_service["build"] = workspace.relpath(
    __file__, "apps/firecrawl/apps/playwright-service-ts")
playwright_service["environment"] = {"PORT": PLAYWRIGHT_SERVICE_PORT}
