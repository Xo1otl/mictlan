import os
import yaml
from relationaldb import *
from util import workspace

# Variable to store the volume mappings for MySQL initialization scripts
mysql_init_script_volumes = []

# Find all .mysql.sql files in folders under build
for filepath in workspace.globrelpaths(__file__, 'build/*/[0-9]*-*.mysql.sql'):
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
config = {
    'services': {
        'mysql': {
            'image': 'mysql:latest',
            'ports': ['3306:3306'],
            'environment': {
                'MYSQL_ROOT_PASSWORD': MYSQL_ROOT_PASSWORD
            },
            # Automate volume creation
            'volumes': mysql_init_script_volumes
        }
    }
}

# Write the configuration to a docker-compose.yaml file
with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
