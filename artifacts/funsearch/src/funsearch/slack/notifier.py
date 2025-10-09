import os
import json
import urllib.request
from typing import List
from funsearch import presenter

try:
    webhook_url = os.environ["SLACK_WEBHOOK_URL"]
except KeyError:
    from .env import WEBHOOK_URL
    webhook_url = WEBHOOK_URL


class SlackNotifier(presenter.ResultNotifier):
    """Slack通知を送信するクラス"""
    
    def send_message(self, message: List[str]) -> bool:
        """
        Webhook経由でSlackにメッセージをブロック形式で送信
        
        Args:
            message: 送信するメッセージのリスト
            
        Returns:
            送信成功時True、失敗時False
        """
        blocks = []
        for msg in message:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": msg
                }
            })
        
        payload = {"blocks": blocks}
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req) as response:
                return response.status == 200
                
        except Exception as e:
            print(f"Slack webhook error: {e}")
            return False