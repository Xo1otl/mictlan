import requests
import streamlit as st

st.set_page_config(
    layout="wide"
)

r"""
### cloudflare tunnelでのデプロイが便利だった

1. どっかでドメインを取得
2. cloudflareに登録
3. cloudflareのチュートリアルに従い権威サーバーをcloudflareに変更
4. 好きなサービスをdockerで建てる、ssl化不要
5. cloudflare zero trust tunnelを開いてサービスと同じネットワーク上にcloudflare tunnelのコンテナを建てる
6. cloudflareのコンソールからホストしたいドメインと、docker network上のアドレスを指定
7. httpsでの公開がこれで完了

### streamlitが超便利だったのでブログ書いてcloudflareでデプロイしたのがこちらのブログ

**なんとlatexがかけます！**

`マクスウェル方程式`
$$
\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}
\end{align}
$$
"""

"""
SEOとかは知らん
"""

"""
👇pythonなので動的生成もできる
"""

# Fetch a random interesting fact from an API
response = requests.get("https://uselessfacts.jsph.pl/random.json?language=en")
if response.status_code == 200:
    fact = response.json().get("text", "No fact available at the moment.")
else:
    fact = "No fact available at the moment."

st.text(f"Did you know? {fact}")
