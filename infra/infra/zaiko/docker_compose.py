from util import workspace
from infra import broker
from infra import nosql
from . import *

package_dir = workspace.relpath(__file__, 'apps/zaiko')

docker_compose = {
    'services': {
        'zaiko': {
            'build': package_dir,
            'volumes': [
                f'./entrypoint.sh:/entrypoint.sh',
                f'./{STOCK_CONNECTOR_FILE}:/{STOCK_CONNECTOR_FILE}',
            ],
            'depends_on': [broker.CONSOLE_CONTAINER_NAME, nosql.CONTAINER_NAME],
            'entrypoint': '/entrypoint.sh',
            'ports': [
                "1234:80"
            ]
        }
    }
}
