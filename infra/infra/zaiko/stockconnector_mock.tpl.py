import yaml
import os

stock_connector = {
    "input": {
        "label": "",
        "kafka": {
            "addresses": ["redpanda:9092"],
            "topics": ["zaiko.stock.projections"],
            "consumer_group": "zaiko.stock.projector",
            "checkpoint_limit": 1024,
            "auto_replay_nacks": True
        }
    },
    "pipeline": {
        "processors": [
            {
                "schema_registry_decode": {
                    "url": "http://redpanda:8081",
                }
            },
            {
                "mutation": "root = this"
            }
        ]
    },
    "output": {
        "stdout": {}
    }
}

target = os.path.join(os.path.dirname(__file__), "stockconnector_mock.yaml")

with open(target, 'w') as file:
    yaml.dump(stock_connector, file)

print(f"[zaiko] stockconnector has been written to {target}.")
