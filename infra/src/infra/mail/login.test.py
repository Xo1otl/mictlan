import imaplib
import ssl
from infra import mail


def fetch_emails():
    try:
        # SSLでIMAPサーバーに接続
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(mail.FQDN, 993, ssl_context=context) as client:
            # ログイン
            client.login(mail.ADMIN_USERNAME, mail.ADMIN_PASSWORD)

            print("Login successful")

            # INBOXを選択
            client.select('INBOX')

            # 未読メールを検索
            _, messages = client.search(None, 'UNSEEN')

            # メールIDが返ってくる（例：b'1 2 3'）
            mail_ids = messages[0].split()

            # 各メールを取得して表示
            for mail_id in mail_ids:
                status, msg_data = client.fetch(mail_id, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        # メールの内容を表示
                        print(response_part[1].decode('utf-8'))

            # ログアウト
            client.logout()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    fetch_emails()
