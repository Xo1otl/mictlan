from relationaldb import *

# volumesの部分自動化できそう
compose = f"""\
services:
  mysql:
    image: mysql:latest
    ports:
      - 3306:3306
    environment:
      MYSQL_ROOT_PASSWORD: {MYSQL_ROOT_PASSWORD}
    volumes:
      - ../koemade/01-initmysqluser.sql:/docker-entrypoint-initdb.d/01-koemade-initmysqluser.sql
      - ../koemade/02-initmysqltables.sql:/docker-entrypoint-initdb.d/02-koemade-initmysqltables.sql
      - ../ossekai/01-initmysqluser.sql:/docker-entrypoint-initdb.d/01-ossekai-initmysqluser.sql
      - ../ossekai/02-initmysqltables.sql:/docker-entrypoint-initdb.d/02-ossekai-initmysqltables.sql
"""

print(compose)
