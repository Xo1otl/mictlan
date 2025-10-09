-- lemmy ユーザーが存在しない場合のみ作成
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'lemmy') THEN
    CREATE USER lemmy WITH PASSWORD 'h71BIDej311Hkal';
  END IF;
END $$;

-- lemmy データベースが存在しない場合のみ作成
-- ここでは DO ブロックを使わずに直接実行する
SELECT 'CREATE DATABASE lemmy OWNER lemmy'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'lemmy')\gexec
