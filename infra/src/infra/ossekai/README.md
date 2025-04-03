# ossekai

データベース空の状態じゃないとentrypoint.shが途中で止まる謎現象がある

# TODO

RAGのためのDBのベクトル化、lanceDB使う

# FIXME

compose upの時postgresの起動より先にanswerが起動し、db接続失敗によりinstallモードになるのが上に書いた謎現象の原因だった

なのでhealth checkを導入する必要がある
