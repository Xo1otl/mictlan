import yaml

config = {
    'services': {
        'elasticsearch': {
            'image': 'docker.elastic.co/elasticsearch/elasticsearch:8.15.2',
            'ports': ['9200:9200'],
            'environment': [
                'discovery.type=single-node',
                'xpack.security.enabled=false'
            ]
        },
        'kibana': {
            'image': 'docker.elastic.co/kibana/kibana:8.14.0',
            'environment': {
                'ELASTICSEARCH_HOSTS': 'http://elasticsearch:9200'
            },
            'ports': ['5601:5601']
        }
    }
}

with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
