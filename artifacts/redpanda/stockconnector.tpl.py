import yaml
import os
from infra import broker
from infra import rdb

DB_NAME = "zaiko"
DB_USER = "zaiko_user"
DB_PASSWORD = "zaiko_password"

mapping_processor = {
    "mapping": """
root.Sub = this.Sub
root.Sales = this.Sales
root.Stocks = this.Stocks.key_values()
root.Args = [this.Sub, this.Sales].concat((root.Stocks.map_each(i -> [this.Sub, i.key, i.value]).flatten()).flatten())
"""
}

# mysqlのmulti queryを使うためには、allowMultiQueries=trueを指定する必要がある
# redpanda connectにはそのオプションがないため、詰んでる
query = """\
SELECT * FROM sales;
SELECT * FROM stocks;
"""

# stored procedureを作成してから、prepared statementで呼び出すことで自由度があがる
# しかしvalues等の可変長引数は無理なので、今回詰んでる
init_statement = """\
CREATE PROCEDURE IF NOT EXISTS upsert_sales_and_stocks(
    IN sub_in VARCHAR(255),
    IN price_in VARCHAR(255),
    IN stocks_values TEXT
)
BEGIN
    START TRANSACTION;

    INSERT INTO sales (sub, price)
    VALUES (sub_in, price_in)
    ON DUPLICATE KEY UPDATE price = price_in;

    DELETE FROM stocks WHERE sub = sub_in;

    SET @insert_sql = CONCAT('INSERT INTO stocks (sub, name, amount) VALUES ', stocks_values, ';');
    
    PREPARE stmt FROM @insert_sql;
    EXECUTE stmt;
    DEALLOCATE PREPARE stmt;

    COMMIT;
END;
"""

# 極論processorを二つ書けば可能だけど、ごり押しすぎるのでrdbは引退
sales_stocks_sql = {
    "label": "sales_stocks_sql",
    "sql_raw": {
        "driver": "mysql",
        "dsn": f"{DB_USER}:{DB_PASSWORD}@tcp({rdb.ADDR})/{DB_NAME}?allowMultiQueries=true",
        "query": query,
        "unsafe_dynamic_query": True,
        "args_mapping": "root = this.Args",
        "init_statement": init_statement
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
        sales_stocks_sql
    ],
    "pipeline": {
        "processors": [
            {
                "schema_registry_decode": {
                    "url": broker.SCHEMA_REGISTRY_URL,
                    "avro_raw_json": True
                }
            },
            mapping_processor,
            {
                "try": [
                    {"resource": sales_stocks_sql["label"]},
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
