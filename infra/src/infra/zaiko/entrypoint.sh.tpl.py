import os
from infra import zaiko

# redpanda-connectという名前の実行ファイルがルートにあった
script = f"""\
#!/bin/sh
/redpanda-connect run /{zaiko.STOCK_CONNECTOR_FILE} &
(/initkafka && /echoserver) &
wait
"""

target = os.path.join(os.path.dirname(__file__), "entrypoint.sh")

with open(target, 'w') as file:
    file.write(script)

print(f"[zaiko] entrypoint.sh has been written to {target}.")
