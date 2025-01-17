import streamlit as st
import uuid
import redis
import time

st.set_page_config(
    page_title="アキネイタースタイルアプリ", page_icon=":thinking_face:", layout="wide"
)


@st.cache_resource
def get_redis_conn():
    try:
        # TODO: これもinfra packageか環境変数から取得する
        r = redis.Redis(host="redis", port=6379, db=0)
        r.ping()
        return r
    except:
        st.error("Redisサーバーに接続できませんでした")
        st.stop()


r = get_redis_conn()

MAX_SESSIONS = 30  # 最大同時セッション数
SESSION_TTL = 60 * 10  # セッションの有効期限 (秒)
REDIS_PREFIX = "akinator_session"


def session_available() -> bool:
    session_id = st.session_state.get("session_id")
    if session_id:
        if r.exists(session_id):
            r.expire(session_id, SESSION_TTL)
            return True
        else:
            del st.session_state.session_id

    active_sessions = r.keys(f"{REDIS_PREFIX}:*")

    if len(active_sessions) < MAX_SESSIONS:  # type: ignore
        new_session_id = f"{REDIS_PREFIX}:{uuid.uuid4()}"
        r.set(new_session_id, time.time())
        r.expire(new_session_id, SESSION_TTL)
        st.session_state.session_id = new_session_id
        return True
    else:
        return False


if not session_available():
    """
    こんにちは！アクセスありがとうございます！

    ただいま混みあっていて、新しいセッションを開始するのが難しい状態です😢

    サーバーの資源に限りがあって、快適にご利用いただくために、同時接続数を制限させていただいています。
    せっかく来ていただいたのに、本当に申し訳ございません。

    少し時間を置いてから、もう一度試してみてください。
    """
    st.stop()


texts, image = st.columns(2)
with texts:
    st.title("アキネイタースタイルアプリへようこそ!")
    """
    このアプリでは、推測ゲームをプレイしたり、知識ベースに貢献したりできます。
    
    **docs**: このアプリの仕組みについて詳細を技術的に説明しています！

    **play:** あなたが考えていることを推測するアプリの能力をテストします。
    
    **train:** 新しい分野、質問、場合に対する回答を追加して、アプリの改善に役立ててください。

    ソースコードはこちら: [GitHub](https://github.com/Xo1otl/mictlan/tree/main/apps/akinator) 
    """
with image:
    st.image("assets/akinator.png", width=400)

"""
--- 
現在、サービスのデータ量が十分でないため、以下の点にご協力をお願いします。

1. **デバッグ情報からケースを選択して遊んでください**
   - 本当は皆さんの好きなケースを当てられるようになりたいのですが、現在は学習データが少ないため、デバッグ情報に載っているケースから選んで遊んでいただけると助かります！

2. **同じケースで何度回答しても学習しない場合があります**
   - 他のケースでの回答データがない場合、同じケースを繰り返しても比較対象がなく、学習が進みません。ぜひ、他のケースも試してみてください！
   
3. **train画面について**
    - 編集機能がないので入力ミスがないかよく確認してください。
    - 新しいケースを追加して、アプリの改善に貢献してくれると大変助かります！
    - 既存の質問では分類不能なケースがある場合に、いい感じにカテゴライズできそうなセンスのいい質問を追加してくれると超絶助かります。

ご理解とご協力をいただけますと幸いです。
"""

# st.write("---")
# st.write("デバッグ情報")
# st.write(f"セッションID: {st.session_state.session_id}")
