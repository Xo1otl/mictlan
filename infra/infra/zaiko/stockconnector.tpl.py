import yaml
import os
from infra import broker

stock_connector = {
    "input": {
        "label": "",
        "kafka": {
            "addresses": [broker.KAFKA_ADDR],
            "topics": ["zaiko.stock.projections"],
            "consumer_group": "zaiko.stock.projector5",
            "checkpoint_limit": 1024,
            "auto_replay_nacks": True
        }
    },
    "pipeline": {
        "processors": [
            {
                "schema_registry_decode": {
                    "url": broker.SCHEMA_REGISTRY_URL,
                }
            },
            {
                "mapping": """|
                root.Sales = this.Sales
                root.Stocks = this.Stocks.key_values().sort_by(pair -> pair.key)
                """
            }
        ]
    },
    "output": {
        "stdout": {}
    }
}

target = os.path.join(os.path.dirname(__file__), "stockconnector.yaml")

with open(target, 'w') as file:
    yaml.dump(stock_connector, file)

print(f"[zaiko] stockconnector has been written to {target}.")
