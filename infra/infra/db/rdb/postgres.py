from . import *
from infra import knowledgebase

postgres_service = {
    'image': 'postgres:16',
    'container_name': 'postgres',
    'restart': 'unless-stopped',
    'volumes': ['./postgres:/var/lib/postgresql/data'],
    'ports': [f'{POSTGRES_PORT}:{POSTGRES_PORT}'],
    'healthcheck': {
        'test': ['CMD-SHELL', f'pg_isready -U {knowledgebase.USERNAME}'],
        'interval': '10s',
        'timeout': '5s',
        'retries': 5
    },
    'environment': {
        'POSTGRES_USER': knowledgebase.USERNAME,
        'POSTGRES_PASSWORD': POSTGRES_PASSWORD,
        'POSTGRES_DB': knowledgebase.DBNAME,
        'PGDATA': '/var/lib/postgresql/data/pgdata'
    }
}
