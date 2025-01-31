from . import *

redis_service = {
    'services': {
        'redis': {
            'image': 'redis',
            'container_name': 'redis',
            'restart': 'unless-stopped',
            'ports': [f'{PORT}:6379'],
            'volumes': ['redis_data:/data'],
            'healthcheck': {
                'test': ['CMD', 'redis-cli', '--raw', 'incr', 'ping'],
                'interval': '10s',
                'timeout': '5s',
                'retries': 5
            }
        }
    },
    'volumes': {
        'redis_data': None
    }
}
