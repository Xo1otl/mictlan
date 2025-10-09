-- affine ユーザーが存在しない場合のみ作成
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'affine') THEN
    CREATE USER affine WITH PASSWORD 'h71BIDej311Hkal';
  END IF;
END $$;

-- affine データベースが存在しない場合のみ作成
-- ここでは DO ブロックを使わずに直接実行する
SELECT 'CREATE DATABASE affine OWNER affine'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'affine')\gexec
