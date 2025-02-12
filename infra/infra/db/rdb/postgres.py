from . import *
from workspace import util, path
import os

postgres_init_script_volumes = ['postgres_data:/var/lib/postgresql/data']

# Find all .mysql.sql files in folders under build
for filepath in util.globrelpaths(path.Path(__file__).dir(), 'infra/infra/*/[0-9]*-*.postgres.sql'):
    folder_name = os.path.basename(os.path.dirname(filepath))
    filename = os.path.basename(filepath)
    # Split the filename to insert folder name
    num_and_description = filename.split('-', 1)
    if len(num_and_description) != 2:
        continue  # Skip files that don't match the pattern
    num = num_and_description[0]
    description = num_and_description[1]
    # Build the destination filename
    dest_filename = f'{num}-{folder_name}-{description}'
    # Build the volume mapping
    src = filepath
    dst = f'/docker-entrypoint-initdb.d/{dest_filename}'
    postgres_init_script_volumes.append(f'{src}:{dst}')


postgres_service = {
    'image': 'postgres:latest',
    'container_name': 'postgres',
    'restart': 'unless-stopped',
    'volumes': postgres_init_script_volumes,
    'ports': [f'{POSTGRES_PORT}:{POSTGRES_PORT}'],
    'healthcheck': {
        'test': ['CMD-SHELL', f'pg_isready -U postgres'],
        'interval': '10s',
        'timeout': '5s',
        'retries': 5
    },
    'environment': {
        'POSTGRES_USER': 'postgres',
        'POSTGRES_PASSWORD': POSTGRES_PASSWORD,
        'PGDATA': '/var/lib/postgresql/data/pgdata'
    }
}
