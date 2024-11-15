#!/usr/bin/env python3

from fssearch import file
from fssearch import searchengine
from infra import searchengine as infra_searchengine
import meilisearch
import argparse

index_name = "fssearch-microsoft_graphrag"


def clear_index():
    """Clear the MeiliSearch index"""
    client = meilisearch.Client(
        'http://meilisearch:7700',
        infra_searchengine.MEILI_MASTER_KEY
    )
    client.index(index_name).delete()
    print("Index cleared successfully")


def index_repository():
    """Index files from the specified directory"""
    collector = file.GitCollector(
        "https://github.com/microsoft/graphrag",
        ignore_patterns=[
            "*.min.css",
            "*-min.css",
            "*.sum",
            "known_hosts",
            "cdk.json",
            "koemadeinfo.html",
            "docs/**",
            "examples/**",
            "examples*/**",
            ".vscode/**",
            ".github/**",
            ".semversioner/**",
            "tests/**",
        ]
    )
    formatter = file.DocumentFormatter()
    processor = file.Processor(collector, formatter)
    uploader = searchengine.MeilisearchUploader(
        "http://meilisearch:7700",
        infra_searchengine.MEILI_MASTER_KEY,
        index_name
    )
    searchengine.Indexer(processor, uploader).index()
    print("Files indexed successfully")


def main():
    parser = argparse.ArgumentParser(description='File Search Indexer')
    parser.add_argument('--clear', action='store_true',
                        help='Clear the existing index before indexing')

    args = parser.parse_args()

    if args.clear:
        clear_index()
        return

    index_repository()


if __name__ == "__main__":
    main()
