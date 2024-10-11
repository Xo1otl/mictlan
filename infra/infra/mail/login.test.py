import imaplib
import ssl
from infra import mail

# 接続情報
IMAP_SERVER = mail.FQDN  # Dovecotサーバーのアドレス
IMAP_PORT = 993  # IMAP over SSL/TLS
USERNAME = 'debug'  # メールアカウントのユーザー名
PASSWORD = 'Abcd1234*'  # メールアカウントのパスワード


def fetch_emails():
    try:
        # SSLでIMAPサーバーに接続
        context = ssl.create_default_context()
        with imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT, ssl_context=context) as mail:
            # ログイン
            mail.login(USERNAME, PASSWORD)

            # INBOXを選択
            mail.select('INBOX')

            # 未読メールを検索
            status, messages = mail.search(None, 'UNSEEN')

            # メールIDが返ってくる（例：b'1 2 3'）
            mail_ids = messages[0].split()

            # 各メールを取得して表示
            for mail_id in mail_ids:
                status, msg_data = mail.fetch(mail_id, '(RFC822)')
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        # メールの内容を表示
                        print(response_part[1].decode('utf-8'))

            # ログアウト
            mail.logout()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    fetch_emails()
