import os
from infra.idp import DBUSER, DBNAME, POSTGRES_PASSWORD

init_sql = f"""\
-- {DBUSER} ユーザーが存在しない場合のみ作成
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '{DBUSER}') THEN
    CREATE USER {DBUSER} WITH PASSWORD '{POSTGRES_PASSWORD}';
  END IF;
END $$;

-- {DBNAME} データベースが存在しない場合のみ作成
-- ここでは DO ブロックを使わずに直接実行する
SELECT 'CREATE DATABASE {DBNAME} OWNER {DBUSER}'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '{DBNAME}')\\gexec
"""

filename = '01-init.postgres.sql'

# 実行しているコンテキストによらずtplと同じ場所に出力
target = os.path.join(os.path.dirname(__file__), filename)

with open(target, 'w') as file:
    file.write(init_sql)

print(f"[idp] {filename} has been written to {target}.")
