import streamlit as st
from typing import Any, Dict, Tuple, Callable
from akinator import qa
import torch

if st.session_state.get("session_id") is None:
    st.switch_page("app.py")


@st.cache_resource
def init_akinator():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


device = init_akinator()
# cache_resourceしたらqueryの結果までキャッシュされてしまうせいでsessionで代用
if st.session_state.get("play_repo") is None:
    st.session_state.play_repo = qa.default_repo()
repo = st.session_state.play_repo
categories = repo.categories()


def init_game(category: str):
    dataset = repo.dataset(category)
    context = qa.Context(dataset, device)
    context.complete()
    return context


initial_state = {
    "categories": categories,
    "category": None,
    "context": None,  # selectorが使用するcontextで、質問や回答の確率計算に必要なテンソルを保持する
    "selector": None,  # contextを使用して質問を選ぶ
    "question": None,  # 現在の質問
    "choices": [],  # 現在のカテゴリでの選択肢一覧
    "history": [],  # 質疑応答の履歴
    "answer": None,  # 確定した回答
}

SELECT_CATEGORY = "SELECT_CATEGORY"  # カテゴリが選択できる
SELECT_CHOICE = "SELECT_CHOICE"
SEND_ANSWER = "SEND_ANSWER"
PLAY_AGAIN = "PLAY_AGAIN"

Action = Tuple[str, Dict[str, Any]]


def check_status(context: qa.Context, top_n: int = 3) -> str | None:
    top_probs, top_indices = torch.topk(
        context.p_case_tensor, min(top_n, len(context.p_case_tensor)))

    print("Current most likely cases:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(
            f"  {i+1}. {context.case_idx_to_id[int(idx.item())]} ({prob.item():.4f})")

    if top_probs[0].item() > 0.7:
        print(
            f"The most likely case is {context.case_idx_to_id[int(top_indices[0].item())]} with probability {top_probs[0].item():.4f}.")
        return context.case_idx_to_id[int(top_indices[0].item())]


def reducer(state: Dict[str, Any], action: Action):
    type, payload = action
    if type == SELECT_CATEGORY:
        context = init_game(payload["category"])
        selector = qa.Selector(context)
        state["category"] = payload["category"]
        state["context"] = context
        state["selector"] = selector
        state["question"] = selector.best_question()
        state["choices"] = context.choice_to_idx.keys()
        state["cases"] = context.case_id_to_idx.keys()
    if type == SELECT_CHOICE:
        # TODO: カテゴリが選択されているか確認
        # answerがないことを確認
        state["selector"].update_context(state["question"], payload["choice"])
        state["history"].append(
            {"question": state["question"], "choice": payload["choice"]})
        if check_status(state["context"]):
            state["answer"] = check_status(state["context"])
            return
        try:
            state["question"] = state["selector"].best_question()
        except ValueError as e:
            state["answer"] = "不明"
        return
    if type == SEND_ANSWER:
        repo.send_answer(state["category"],
                         payload["answer"], state["history"])
        return
    if type == PLAY_AGAIN:
        # reset to initial state
        state.update(initial_state)
        return


def usePlayer() -> Tuple[Dict[str, Any], Callable[[Action], None]]:
    if "play_state" not in st.session_state:
        st.session_state.play_state = initial_state

    def dispatch(action: Action):
        try:
            reducer(st.session_state.play_state, action)
        except ValueError as e:
            st.error(e)
    return st.session_state.play_state, dispatch


state, dispatch = usePlayer()

st.title("あそぶ")

if state["category"] is None:
    category = st.selectbox(
        "どのジャンルで遊びますか？",
        state["categories"],
        key="play_category_select",
    )
    if st.button("ゲームスタート"):
        dispatch((SELECT_CATEGORY, {"category": category}))
        st.rerun()
    st.stop()
if state["answer"] is not None:
    if state["answer"] == "不明":
        msg1 = "回答が見つかりませんでした"
        msg2 = "正しい答えを選択してください。選択肢にない場合は、trainページから追加してください。"
        default_index = 0
    else:
        msg1 = "正解でしたか？"
        msg2 = "間違っていた場合は正しい答えを選択してください。選択肢にない場合は、trainページから追加してください。"
        default_index = list(state["cases"]).index(state["answer"])
    correct_case = st.selectbox(
        msg1,
        state["cases"],
        key="correct_case",
        index=default_index
    )
    st.write(msg2)
    if st.button("回答を送信", key="send_correct_case"):
        dispatch((SEND_ANSWER, {"answer": correct_case}))
        dispatch((PLAY_AGAIN, {}))
        st.rerun()
    """
    同じ回答や違う回答で繰り返しあそぶと、リアルタイムにどんどん賢くなります！回答を送信したら Let's リトライ！
    """
else:
    st.write(state["question"])
    choice = st.radio("回答を選択", state["choices"], horizontal=True)
    if st.button("回答を確定"):
        dispatch((SELECT_CHOICE, {"choice": choice}))
        st.rerun()
if st.button("はじめから"):
    dispatch((PLAY_AGAIN, {}))
    st.rerun()

st.write("---")
st.write("デバッグ情報")
st.write(state)
