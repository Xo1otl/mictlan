# chat

LibreChatかragflow等を使う

## 設計

- LibreChat
- ollama
- meilisearch
- mongodb
- vectordb
- rag_api, ragflow等

## TODO

- ollamaのコンテナやっぱ分けようかなと思う、`cap add gpu`するコンテナ複数あるけど、多分大丈夫な気がする
- ollamaとuiだけここに所属して、vectordbやmongoやmeilisearchはほかでも再利用できる
