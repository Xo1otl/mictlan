import streamlit as st

"""
# Auth

OAuth2.0 OIDC FIDO2のpasskey認証しかかたんということを学んだのでメモ

サイトを作る上で、認証機能は必須だけど、フルスクラッチは超大変

信頼できる認証局を作り、ユーザーはその認証局でログインを行い、トークンを取得する

公開鍵暗号を用いて、秘密鍵で署名されたトークンを使用し、それぞれのサービスでは認証局が公開している公開鍵で検証を行うようにする

すると、それぞれのサービスはステートレスに認証を行うことが可能

この考え方をもとに、細かいインシデントに対応するためにいろいろ頑張った認証方式がOIDC

また、最近のデバイスは大体指紋認証等を備えており、FIDO2 passkey認証というのが主流になってきているらしい

今後、mictlan内のすべてのサービスはOIDCのrelaying partyとして扱い、さらにパスワードレス認証のみを行うこととする
"""
