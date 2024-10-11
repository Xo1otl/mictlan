import meilisearch
from infra import searchengine
import json

client = meilisearch.Client(
    'http://meilisearch:7700', searchengine.MEILI_MASTER_KEY
)

json_file = open('movies.json', encoding='utf-8')
movies = json.load(json_file)
client.index('movies').add_documents(movies)
