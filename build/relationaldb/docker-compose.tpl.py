import yaml
from relationaldb import *

# List of SQL scripts to map
volume_mappings = [
    ('../koemade/01-initmysqluser.sql',
     '/docker-entrypoint-initdb.d/01-koemade-initmysqluser.sql'),
    ('../koemade/02-initmysqltables.sql',
     '/docker-entrypoint-initdb.d/02-koemade-initmysqltables.sql'),
    ('../ossekai/01-initmysqluser.sql',
     '/docker-entrypoint-initdb.d/01-ossekai-initmysqluser.sql'),
    ('../ossekai/02-initmysqltables.sql',
     '/docker-entrypoint-initdb.d/02-ossekai-initmysqltables.sql')
]

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
            'volumes': [f'{src}:{dst}' for src, dst in volume_mappings]
        }
    }
}

# Write the configuration to a docker-compose.yaml file
with open('docker-compose.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)
