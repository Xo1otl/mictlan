from typing import Protocol, List, Dict, Generator, Union, Iterator, Any


class Bot(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> Generator[Dict[str, Any], None, None]:
        """
        メッセージ履歴を受け取り、応答をストリーミングで返す
        """
        ...


class Repository(Protocol):
    def save(self, user_id: str, chat_id: str, messages: List[Dict[str, str]]) -> None:
        """
        会話を永続化する
        """
        ...


class Manager:
    def __init__(self, bot: Bot, repository: Repository):
        self.bot = bot
        self.repository = repository

    def mediate(
        self,
        user_id: str,
        chat_id: str,
        messages: List[dict]
    ) -> Union[Dict[str, Any], Generator, Iterator]:
        """
        新しいメッセージを処理し、応答をストリーミングで返す

        Args:
            user_id: ユーザーID
            chat_id: チャットID
            messages: メッセージ履歴

        Returns:
            Union[str, Generator, Iterator]: API応答のストリーム
        """

        # ストリーミングレスポンスを生成
        response_stream = self.bot.chat(messages)

        return self._stream_and_save(response_stream, user_id, chat_id, messages)

    def _stream_and_save(
        self,
        response_stream: Generator[Dict[str, Any], None, None],
        user_id: str,
        chat_id: str,
        messages: List[dict],
    ) -> Generator[Dict[str, Any], None, None]:
        """
        レスポンスをストリームしながら内容を蓄積し、完了後に保存する
        """
        full_response = ""

        for chunk in response_stream:
            if chunk.get('message', {}).get('content'):
                content = chunk['message']['content']
                full_response += content
            yield chunk

        # ストリーミング完了後、履歴を保存
        assistant_message = {
            "role": "assistant",
            "content": full_response,
        }

        messages.append(assistant_message)
        self.repository.save(user_id, chat_id, messages)
