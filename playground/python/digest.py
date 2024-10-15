import requests
from requests.auth import HTTPDigestAuth

# 認証情報
username = 'test'
password = 'test'

# 最初のリクエスト先
url = 'http://localhost:3030'

# セッションを作成
session = requests.Session()

# セッションにDigest認証を設定
session.auth = HTTPDigestAuth(username, password)

# Digest認証を使って最初のリクエストを行い、認証情報を保持
response = session.get(url)

# レスポンスの確認
print(f'First request status: {response.status_code}')
print(f'First request content: {response.content.decode()}')

# 同じセッションで再度リクエストを送る（Digest認証は自動的に再利用される）
for i in range(3):
    response = session.get(url)
    print(f'Request {i+1} status: {response.status_code}')
    print(f'Request {i+1} content: {response.content.decode()}')
