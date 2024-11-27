from . import *
from infra.db import rdb

affine_service = {
    'image': 'ghcr.io/toeverything/affine-graphql:stable',
    'container_name': 'affine',
    'command': ['sh', '-c', 'node ./scripts/self-host-predeploy && node ./dist/index.js'],
    'ports': [f'{PORTS[0]}:3010', f'{PORTS[1]}:5555'],
    'depends_on': {
        'redis': {'condition': 'service_healthy'},
        'postgres': {'condition': 'service_healthy'}
    },
    'volumes': [
        './.affine/config:/root/.affine/config',
        './.affine/storage:/root/.affine/storage'
    ],
    'logging': {
        'driver': 'json-file',
        'options': {'max-size': '1000m'}
    },
    'restart': 'unless-stopped',
    'environment': [
        'NODE_OPTIONS="--import=./scripts/register.js"',
        'AFFINE_CONFIG_PATH=/root/.affine/config',
        'REDIS_SERVER_HOST=redis',
        f'DATABASE_URL=postgres://{USERNAME}:{rdb.POSTGRES_PASSWORD}@{rdb.POSTGRES_ADDR}/{DBNAME}',
        'NODE_ENV=production',
        # 'TELEMETRY_ENABLE=false'  # テレメトリ無効化する場合
    ]
}
