compose = f"""\
services:
  redpanda:
    command:
      - redpanda
      - start
      - --kafka-addr internal://0.0.0.0:9092,external://0.0.0.0:19092
      # Address the broker advertises to clients that connect to the Kafka API.
      # Use the internal addresses to connect to the Redpanda brokers'
      # from inside the same Docker network.
      # Use the external addresses to connect to the Redpanda brokers'
      # from outside the Docker network.
      - --advertise-kafka-addr internal://redpanda:9092,external://localhost:19092
      - --pandaproxy-addr internal://0.0.0.0:8082,external://0.0.0.0:18082
      # Address the broker advertises to clients that connect to the HTTP Proxy.
      - --advertise-pandaproxy-addr internal://redpanda:8082,external://localhost:18082
      - --schema-registry-addr internal://0.0.0.0:8081,external://0.0.0.0:18081
      # Redpanda brokers use the RPC API to communicate with each other internally.
      - --rpc-addr redpanda:33145
      - --advertise-rpc-addr redpanda:33145
      # Mode dev-container uses well-known configuration properties for development in containers.
      - --mode dev-container
      # Tells Seastar (the framework Redpanda uses under the hood) to use 1 core on the system.
      - --smp 1
      - --default-log-level=info
    image: docker.redpanda.com/redpandadata/redpanda:latest
    container_name: redpanda
    volumes:
      - redpanda:/var/lib/redpanda/data
    ports:
      - 18081:18081
      - 18082:18082
      - 19092:19092
      - 19644:9644

  redpanda-console:
    container_name: redpanda-console
    image: docker.redpanda.com/redpandadata/console:latest
    entrypoint: /bin/sh
    command: -c 'echo "$$CONSOLE_CONFIG_FILE" > /tmp/config.yml; /app/console'
    environment:
      CONFIG_FILEPATH: /tmp/config.yml
      CONSOLE_CONFIG_FILE: |
        kafka:
          brokers: ["redpanda:9092"]
          schemaRegistry:
            enabled: true
            urls: ["http://redpanda:8081"]
        redpanda:
          adminApi:
            enabled: true
            urls: ["http://redpanda:9644"]
    ports:
      - 8080:8080
    depends_on:
      - redpanda

volumes:
  redpanda: null
"""

with open('docker-compose.yaml', 'w') as file:
    file.write(compose)
