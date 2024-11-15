from util import workspace
from infra import broker
from infra.db import documentdb
from . import *

package_dir = workspace.relpath(__file__, 'apps/zaiko')

compose = {
    'services': {
        'zaiko': {
            'build': package_dir,
            'volumes': [
                f'./entrypoint.sh:/entrypoint.sh',
                f'./{STOCK_CONNECTOR_FILE}:/{STOCK_CONNECTOR_FILE}',
            ],
            'depends_on': [broker.CONSOLE_SERVICE_NAME, documentdb.SERVICE_NAME],
            'entrypoint': '/entrypoint.sh',
            'ports': [
                "1234:80"
            ]
        }
    }
}
