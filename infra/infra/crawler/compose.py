import yaml
from util import workspace

yaml_data = workspace.read("apps/firecrawl/docker-compose.yaml")
compose_data = yaml.safe_load(yaml_data)
print(compose_data)
