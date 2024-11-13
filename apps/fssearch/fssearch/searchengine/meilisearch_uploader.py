import meilisearch


class MeilisearchUploader:
    def __init__(self, host: str, api_key: str, index_name: str):
        self.client = meilisearch.Client(host, api_key)
        self.index = self.client.index(index_name)

        # 検索可能なフィールドとフィルタリング属性を設定
        self.index.update_settings({
            'searchableAttributes': [
                'content',
                'filepath'
            ],
            'filterableAttributes': [
                'ext'
            ]
        })

    def upload(self, documents: list[dict]) -> None:
        """ドキュメントをMeilisearchにアップロードする"""
        if documents:
            self.index.add_documents(documents)
            print(f"Indexed {len(documents)} files successfully")
        else:
            print("No text files found to index")
