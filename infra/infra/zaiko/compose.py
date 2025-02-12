from infra import broker
from infra.db import documentdb
from . import *
from workspace import path

package_dir = str(path.Path("apps/zaiko").rel2(path.Path(__file__).dir()))

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
                "1235:80"
            ]
        }
    }
}
