import os
from workspace import util, path
from . import *

# Variable to store the volume mappings for MySQL initialization scripts
mysql_init_script_volumes = [
    'mysql_data:/var/lib/mysql'
]

paths = util.globpaths('infra/infra/*/[0-9]*-*.mysql.sql')
paths = [p.rel2(path.Path(__file__).dir()) for p in paths]
# Find all .mysql.sql files in folders under build
for filepath in paths:
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
    mysql_init_script_volumes.append(f'{src}:{dst}')

# Build the config object
mysql_service = {
    'image': 'mysql:latest',
    'ports': [f'{MYSQL_PORT}:{MYSQL_PORT}'],
    'environment': {
        'MYSQL_ROOT_PASSWORD': MYSQL_ROOT_PASSWORD
    },
    # Automate volume creation
    'volumes': mysql_init_script_volumes
}
