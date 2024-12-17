import streamlit as st

st.set_page_config(
    page_title="アキネイタースタイルアプリ",
    page_icon=":thinking_face:",
    layout="wide"
)

st.title("アキネイタースタイルアプリへようこそ!")
st.write("""
    このアプリでは、推測ゲームをプレイしたり、知識ベースに貢献したりできます。

    **play:** あなたが考えていることを推測するアプリの能力をテストします。

    **train:** 新しい分野、質問、場合に対する回答を追加して、アプリの改善に役立ててください。
    """)
