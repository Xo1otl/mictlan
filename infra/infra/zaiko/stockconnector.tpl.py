import yaml
import os
from infra import broker
from infra import zaiko
from infra import nosql

sales_stocks_mongo = {
    "label": "sales_stocks_mongo",
    "mongodb": {
        "url": f"mongodb://{nosql.CONTAINER_NAME}:{nosql.PORT}",
        "database": zaiko.DB_NAME,
        "collection": "sales_stocks",
        "operation": "replace-one",
        "write_concern": {
            "w": "majority",
            "j": True
        },
        "upsert": True,
        "document_map": """
root.Sub = this.Sub
root.Stocks = this.Stocks
root.Sales = this.Sales
""",
        "filter_map": "root.Sub = this.Sub"
    }
}

stock_connector = {
    "input": {
        "kafka_franz": {
            "seed_brokers": [broker.KAFKA_ADDR],
            "topics": ["zaiko.stock.projections"],
            "consumer_group": "zaiko.stock.projector18",
            "auto_replay_nacks": True,
        }
    },
    "processor_resources": [
        sales_stocks_mongo
    ],
    "pipeline": {
        "processors": [
            {
                "schema_registry_decode": {
                    "url": broker.SCHEMA_REGISTRY_URL,
                }
            },
            {
                "try": [
                    {"resource": sales_stocks_mongo["label"]},
                ]
            },
            {
                "catch": [{
                    "log": {
                        "message": "Processing failed due to: ${!error()}"
                    }
                }]
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
