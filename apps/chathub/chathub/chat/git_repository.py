import os
import json
import git
from typing import Dict, List


class GitRepository:
    def __init__(self, base_path: str = "./repositories"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def save(self, user_id: str, chat_id: str, messages: List[Dict[str, str]]) -> None:
        # Initialize user repository
        repo_path = os.path.join(self.base_path, user_id)
        if not os.path.exists(repo_path):
            os.makedirs(repo_path)
            repo = git.Repo.init(repo_path)
        else:
            repo = git.Repo(repo_path)

        # Save messages
        file_path = os.path.join(repo_path, f"{chat_id}.json")
        with open(file_path, 'w') as f:
            json.dump({"messages": messages}, f, indent=2, ensure_ascii=False)

        # Commit
        repo.index.add([f"{chat_id}.json"])
        repo.index.commit(f'Update chat {chat_id}')