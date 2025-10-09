-- authelia ユーザーが存在しない場合のみ作成
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'authelia') THEN
    CREATE USER authelia WITH PASSWORD 'aoihva39hAA7oauhHG1';
  END IF;
END $$;

-- authelia データベースが存在しない場合のみ作成
-- ここでは DO ブロックを使わずに直接実行する
SELECT 'CREATE DATABASE authelia OWNER authelia'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'authelia')\gexec
