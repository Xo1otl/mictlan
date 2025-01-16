import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import configparser


def send_email(from_email, from_name, to_email, smtp_host, smtp_port, smtp_username, smtp_password, encryption='tls'):
    # メールの内容を作成
    subject = "テストメール"
    body = "これはAWS SES SMTPを使用したテストメールです。"

    # メールのヘッダーと本文を設定
    msg = MIMEMultipart()
    msg['From'] = f"{from_name} <{from_email}>"
    msg['To'] = to_email
    msg['Subject'] = Header(subject, 'utf-8')  # type: ignore

    # 本文を追加
    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        # SMTPサーバーに接続
        if encryption == 'tls':
            server = smtplib.SMTP(smtp_host, smtp_port)
            server.starttls()  # TLS暗号化を開始
        elif encryption == 'ssl':
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            raise ValueError("Encryption must be either 'tls' or 'ssl'")

        # SMTPサーバーにログイン
        server.login(smtp_username, smtp_password)

        # メールを送信
        server.sendmail(from_email, to_email, msg.as_string())
        print("メールが正常に送信されました。")

    except Exception as e:
        print(f"メール送信に失敗しました: {e}")

    finally:
        # SMTPサーバーとの接続を閉じる
        server.quit()  # type: ignore


def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)

    # 設定を読み込む
    smtp_config = {
        'from_email': config['SMTP']['from_email'],
        'from_name': config['SMTP']['from_name'],
        'to_email': config['SMTP']['to_email'],
        'smtp_host': config['SMTP']['smtp_host'],
        'smtp_port': int(config['SMTP']['smtp_port']),
        'smtp_username': config['SMTP']['smtp_username'],
        'smtp_password': config['SMTP']['smtp_password'],
        'encryption': config['SMTP']['encryption']
    }

    print(smtp_config)

    return smtp_config


# 設定ファイルを読み込む
config_file = 'config.ini'
smtp_config = load_config(config_file)

# メールを送信
send_email(**smtp_config)
