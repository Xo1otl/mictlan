import streamlit as st
from typing import Dict, Tuple, Any, Callable
from akinator import train
import uuid


def generate_uuid() -> str:
    """UUIDを生成する関数"""
    return str(uuid.uuid4())


@st.cache_resource
def init_akinator():
    """Akinatorの初期化関数 (最初の1回だけ実行)"""
    repo = train.default_repo()
    categories = repo.categories()
    return repo, categories


repo, categories = init_akinator()

initial_state = {
    "category_id": None,
    "question_id": None,
    "case_id": None,
    "choice": None,
    "categories": categories,
}

SELECT_CATEGORY = "SELECT_CATEGORY"
ADD_CATEGORY = "ADD_CATEGORY"
EDIT_CHOICES = "EDIT_CHOICES"
SELECT_QUESTION = "SELECT_QUESTION"
ADD_QUESTION = "ADD_QUESTION"
SELECT_CASE = "SELECT_CASE"
ADD_CASE = "ADD_CASE"
SELECT_CHOICE = "SELECT_CHOICE"

Action = Tuple[str, Dict[str, Any]]


def reducer(state: Dict[str, Any], action: Action) -> Dict[str, Any]:
    type, payload = action
    if type == SELECT_CATEGORY:
        state["category_id"] = payload["category_id"]
        state["question_id"] = None
        state["case_id"] = None
        state["choice"] = None
        return state

    if type == ADD_CATEGORY:
        if any(
            category_data["text"] == payload["category_name"]
            for category_data in state["categories"].values()
        ):
            raise ValueError("そのカテゴリ名はすでに存在します")

        category_id = repo.add_category(payload["category_name"])
        state["categories"][category_id] = {
            "text": payload["category_name"],
            "choices": [],
            "cases": {},
            "questions": {},
        }
        state["category_id"] = category_id
        state["question_id"] = None
        state["case_id"] = None
        state["choice"] = None
        return state

    if state["category_id"] is None:
        raise ValueError("カテゴリが選択されていません")

    if type == EDIT_CHOICES:
        if (
            state["categories"][state["category_id"]]["questions"]
            or state["categories"][state["category_id"]]["cases"]
        ):
            raise ValueError("質問または場合が存在するため、選択肢を編集できません")

        # choicesが空文字列を含む場合、エラーを出す
        if len(payload["choices"]) == 0:
            raise ValueError("選択肢が空です")
        if any(not choice.strip() for choice in payload["choices"]):
            raise ValueError("選択肢に空文字列が含まれています")

        repo.edit_choices(state["category_id"], payload["choices"])
        state["categories"][state["category_id"]
                            ]["choices"] = payload["choices"]
        return state

    if type == SELECT_QUESTION:
        state["question_id"] = payload["question_id"]
        state["choice"] = None
        return state

    if type == ADD_QUESTION:
        if any(
            question_data["text"] == payload["question_text"]
            for question_data in state["categories"][state["category_id"]]["questions"].values()
        ):
            raise ValueError("その質問はすでに存在します")

        question_id = repo.add_question(
            state["category_id"], payload["question_text"])
        state["categories"][state["category_id"]]["questions"][question_id] = {
            "text": payload["question_text"]
        }
        state["question_id"] = question_id
        state["choice"] = None
        return state

    if type == SELECT_CASE:
        state["case_id"] = payload["case_id"]
        state["choice"] = None
        return state

    if type == ADD_CASE:
        if any(
            case_data["text"] == payload["case_name"]
            for case_data in state["categories"][state["category_id"]]["cases"].values()
        ):
            raise ValueError("その場合はすでに存在します")

        case_id = repo.add_case(state["category_id"], payload["case_name"])
        state["categories"][state["category_id"]]["cases"][case_id] = {
            "text": payload["case_name"]
        }
        state["case_id"] = case_id
        state["choice"] = None
        return state

    if type == SELECT_CHOICE:
        if state["question_id"] is None or state["case_id"] is None:
            raise ValueError("質問と場合が選択されていません")
        if payload["choice"] not in state["categories"][state["category_id"]]["choices"]:
            raise ValueError("その回答は選択肢に存在しません")

        repo.add_answer(
            state["category_id"],
            state["question_id"],
            state["case_id"],
            payload["choice"],
        )
        state["choice"] = payload["choice"]
        return state

    raise ValueError(f"Unknown action type: {type}")


def useTrainer() -> Tuple[Dict[str, Any], Callable[[Action], None]]:
    if "state" not in st.session_state:
        st.session_state.state = initial_state

    def dispatch(action: Action) -> None:
        # streamlitはstateによる画面のレンダリングは機能はないのでmutableでよい
        try:
            st.session_state.state = reducer(st.session_state.state, action)
        except ValueError as e:
            st.error(e)
    return st.session_state.state, dispatch


state, dispatch = useTrainer()

st.title("学習する")

category_tab, question_tab, case_tab, answer_tab = st.tabs(
    ["カテゴリ", "質問", "場合", "回答"])

with category_tab:
    st.header("カテゴリを選択または追加")

    category_id_to_text = {}
    category_ids = []
    for category_id, category_data in state["categories"].items():
        category_id_to_text[category_id] = category_data["text"]
        category_ids.append(category_id)

    st.selectbox(
        "カテゴリを選択",
        options=category_ids,
        format_func=lambda id: category_id_to_text.get(id, ""),
        index=None if state["category_id"] is None else category_ids.index(
            state["category_id"]),
        placeholder="既存のカテゴリを入力してください",
        key="category_select",
        on_change=lambda: dispatch(
            (SELECT_CATEGORY, {"category_id": st.session_state.get("category_select", "")})),
    )

    new_category_name = st.text_input(
        "カテゴリを追加",
        placeholder="新しいカテゴリを入力してください",
        key="new_category_input",
    )
    if new_category_name:
        if new_category_name not in category_id_to_text.values():
            st.subheader(f"{new_category_name}の選択肢を設定")
            options_text = st.text_area("選択肢を改行で区切って入力", value="\n".join([]))
            new_category_options = [
                line.strip() for line in options_text.split("\n") if line.strip()
            ]
            if st.button("選択肢を保存"):
                dispatch((ADD_CATEGORY, {"category_name": new_category_name}))
                dispatch((EDIT_CHOICES, {"choices": new_category_options}))

with question_tab:
    st.header("質問を選択または追加")
    if state["category_id"]:
        category_data = state["categories"][state["category_id"]]

        question_id_to_text = {}
        question_ids = []
        for question_id, question_data in category_data["questions"].items():
            question_id_to_text[question_id] = question_data["text"]
            question_ids.append(question_id)

        selected_question_id = st.selectbox(
            f"{category_data['text']} の質問を選択",
            options=question_ids,
            format_func=lambda id: question_id_to_text.get(
                id, "") if id is not None else "",
            index=None if state["question_id"] is None else question_ids.index(
                state["question_id"]),
            placeholder="既存の質問を入力してください",
            key="question_select",
            on_change=lambda: dispatch(
                (SELECT_QUESTION, {"question_id": st.session_state.get("question_select", "")})),
        )

        st.text_input(
            f"{category_data['text']} の質問を追加",
            placeholder="新しい質問を入力してください",
            key="new_question_input",
            on_change=lambda: dispatch(
                (ADD_QUESTION, {"question_text": st.session_state.new_question_input}))
        )
    else:
        st.warning("カテゴリを選択してください")

with case_tab:
    st.header("場合を選択または追加")
    if state["category_id"]:
        category_data = state["categories"][state["category_id"]]

        case_id_to_text = {}
        case_ids = []
        for case_id, case_data in category_data["cases"].items():
            case_id_to_text[case_id] = case_data["text"]
            case_ids.append(case_id)

        selected_case_id = st.selectbox(
            f"{category_data['text']} の場合を選択",
            options=case_ids,
            format_func=lambda id: case_id_to_text.get(
                id, "") if id is not None else "",
            index=None if state["case_id"] is None else case_ids.index(
                state["case_id"]),
            placeholder="既存の場合を入力してください",
            key="case_select",
            on_change=lambda: dispatch(
                (SELECT_CASE, {"case_id": st.session_state.get("case_select", "")})),
        )

        st.text_input(
            f"{category_data['text']} の場合を追加",
            placeholder="新しい場合を入力してください",
            key="new_case_input",
            on_change=lambda: dispatch(
                (ADD_CASE, {"case_name": st.session_state.new_case_input}))
        )

    else:
        st.warning("カテゴリを選択してください")

with answer_tab:
    st.header("回答を追加")
    if not state["category_id"]:
        st.warning("カテゴリを選択してください")
    if not state["question_id"]:
        st.warning("質問を選択してください")
    if not state["case_id"]:
        st.warning("場合を選択してください")
    if state["category_id"] and state["question_id"] and state["case_id"]:
        category_data = state["categories"][state["category_id"]]
        st.write(f"カテゴリ: **{category_data['text']}**")
        st.write(
            f"質問: **{category_data['questions'][state['question_id']]['text']}**")
        st.write(f"場合: **{category_data['cases'][state['case_id']]['text']}**")
        answer = st.radio(
            "回答を選択", options=category_data["choices"], horizontal=True)
        if st.button("回答を追加"):
            dispatch((SELECT_CHOICE, {"choice": answer}))
            st.success(
                f"カテゴリ: {category_data['text']}, 場合: {category_data['cases'][state['case_id']]['text']}, 質問: {category_data['questions'][state['question_id']]['text']} に対して「{answer}」と回答しました")

st.write("---")
st.write("デバッグ情報")
st.write(state)
