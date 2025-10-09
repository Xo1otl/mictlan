from typing import Protocol, List


class ResultNotifier(Protocol):
    """結果通知を送信するプロトコル"""

    def send_message(self, message: List[str]) -> bool:
        """
        メッセージを送信する

        Args:
            message: 送信するメッセージのリスト

        Returns:
            送信成功時True、失敗時False
        """
        ...
