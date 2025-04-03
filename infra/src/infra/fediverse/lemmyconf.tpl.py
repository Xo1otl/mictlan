import os
from infra.fediverse import DBUSER, DBNAME, DBPASSWORD, DOMAIN, SMTP_SERVER, SMTP_DOMAIN, SMTP_PASSWORD, SMTP_EMAIL
from infra.db import rdb
from infra import mail

lemmy_hjson = f"""\
{{
  # for more info about the config, check out the documentation
  # https://join-lemmy.org/docs/en/administration/configuration.html

  database: {{
    uri: "postgresql://{DBUSER}:{DBPASSWORD}@{rdb.POSTGRES_ADDR}/{DBNAME}"
  }}
  hostname: "{DOMAIN}"
  pictrs: {{
    url: "http://pictrs:8080/"
    api_key: "{DBPASSWORD}"
  }}
  email: {{
    smtp_server: "{SMTP_SERVER}:587"
    smtp_login: "{SMTP_EMAIL}"
    smtp_from_address: "xolotl@{SMTP_DOMAIN}"
    smtp_password: "{SMTP_PASSWORD}"
    tls_type: "starttls"
  }}
}}
"""

filename = 'lemmy.hjson'

# 実行しているコンテキストによらずtplと同じ場所に出力
target = os.path.join(os.path.dirname(__file__), filename)

with open(target, 'w') as file:
    file.write(lemmy_hjson)

print(f"[fediverse] {filename} has been written to {target}.")
