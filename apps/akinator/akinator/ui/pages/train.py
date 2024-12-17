import streamlit as st
from typing import Dict, Tuple, Any, Callable
import uuid


def generate_uuid() -> str:
    """UUIDを生成する関数"""
    return str(uuid.uuid4())


mock_data: Dict[str, Dict[str, Any]] = {
    generate_uuid(): {
        "text": "動物",
        "choices": ["はい", "多分はい", "わからない", "多分いいえ", "いいえ"],
        "cases": {
            generate_uuid(): {"text": "カエル"},
            generate_uuid(): {"text": "犬"},
            generate_uuid(): {"text": "猫"},
        },
        "questions": {
            generate_uuid(): {"text": "肉食ですか？"},
            generate_uuid(): {"text": "飛びますか？"},
            generate_uuid(): {"text": "4本足ですか？"},
        },
    },
    generate_uuid(): {
        "text": "乗り物",
        "choices": ["そう", "部分的にそう", "無関係", "いいえ"],
        "cases": {
            generate_uuid(): {"text": "飛行機"},
            generate_uuid(): {"text": "車"},
            generate_uuid(): {"text": "自転車"},
        },
        "questions": {
            generate_uuid(): {"text": "高いですか？"},
            generate_uuid(): {"text": "速いですか？"},
            generate_uuid(): {"text": "肉食ですか？"},
        },
    },
    generate_uuid(): {
        "text": "食べ物",
        "choices": ["はい", "いいえ"],
        "cases": {
            generate_uuid(): {"text": "りんご"},
            generate_uuid(): {"text": "みかん"},
            generate_uuid(): {"text": "バナナ"},
        },
        "questions": {
            generate_uuid(): {"text": "甘いですか？"},
            generate_uuid(): {"text": "赤いですか？"},
            generate_uuid(): {"text": "丸いですか？"},
        },
    },
}

initial_state = {
    "genre_id": None,
    "question_id": None,
    "case_id": None,
    "choice": None,
    "data": mock_data,
}

SELECT_GENRE = "SELECT_GENRE"
ADD_GENRE = "ADD_GENRE"
EDIT_CHOICES = "EDIT_CHOICES"
SELECT_QUESTION = "SELECT_QUESTION"
ADD_QUESTION = "ADD_QUESTION"
SELECT_CASE = "SELECT_CASE"
ADD_CASE = "ADD_CASE"
SELECT_CHOICE = "SELECT_CHOICE"

Action = Tuple[str, Dict[str, Any]]


def reducer(state: Dict[str, Any], action: Action) -> Dict[str, Any]:
    type, payload = action
    if type == SELECT_GENRE:
        state["genre_id"] = payload["genre_id"]
        state["question_id"] = None
        state["case_id"] = None
        state["choice"] = None
        return state

    if type == ADD_GENRE:
        if any(
            genre_data["text"] == payload["genre_text"]
            for genre_data in state["data"].values()
        ):
            raise ValueError("そのジャンル名はすでに存在します")

        genre_id = generate_uuid()
        state["data"][genre_id] = {
            "text": payload["genre_text"],
            "choices": [],
            "cases": {},
            "questions": {},
        }
        state["genre_id"] = genre_id
        state["question_id"] = None
        state["case_id"] = None
        state["choice"] = None
        return state

    if state["genre_id"] is None:
        raise ValueError("分野が選択されていません")

    if type == EDIT_CHOICES:
        if (
            state["data"][state["genre_id"]]["questions"]
            or state["data"][state["genre_id"]]["cases"]
        ):
            raise ValueError("質問または場合が存在するため、選択肢を編集できません")

        # choicesが空文字列を含む場合、エラーを出す
        if len(payload["choices"]) == 0:
            raise ValueError("選択肢が空です")
        if any(not choice.strip() for choice in payload["choices"]):
            raise ValueError("選択肢に空文字列が含まれています")

        state["data"][state["genre_id"]]["choices"] = payload["choices"]
        return state

    if type == SELECT_QUESTION:
        state["question_id"] = payload["question_id"]
        state["choice"] = None
        return state

    if type == ADD_QUESTION:
        if any(
            question_data["text"] == payload["question_text"]
            for question_data in state["data"][state["genre_id"]]["questions"].values()
        ):
            raise ValueError("その質問はすでに存在します")

        question_id = generate_uuid()
        state["data"][state["genre_id"]]["questions"][question_id] = {
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
            case_data["text"] == payload["case_text"]
            for case_data in state["data"][state["genre_id"]]["cases"].values()
        ):
            raise ValueError("その場合はすでに存在します")

        case_id = generate_uuid()
        state["data"][state["genre_id"]]["cases"][case_id] = {
            "text": payload["case_text"]
        }
        state["case_id"] = case_id
        state["choice"] = None
        return state

    if type == SELECT_CHOICE:
        if state["question_id"] is None or state["case_id"] is None:
            raise ValueError("質問と場合が選択されていません")
        if payload["choice"] not in state["data"][state["genre_id"]]["choices"]:
            raise ValueError("その回答は選択肢に存在しません")

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


st.set_page_config(layout="wide")

state, dispatch = useTrainer()

st.title("学習する")

genre_tab, question_tab, case_tab, answer_tab = st.tabs(
    ["分野", "質問", "場合", "回答"])

with genre_tab:
    st.header("分野を選択または追加")

    genre_id_to_text = {}
    genre_ids = []
    for genre_id, genre_data in state["data"].items():
        genre_id_to_text[genre_id] = genre_data["text"]
        genre_ids.append(genre_id)

    st.selectbox(
        "分野を選択",
        options=genre_ids,
        format_func=lambda id: genre_id_to_text.get(id, ""),
        index=None if state["genre_id"] is None else genre_ids.index(
            state["genre_id"]),
        placeholder="既存の分野を入力してください",
        key="genre_select",
        on_change=lambda: dispatch(
            (SELECT_GENRE, {"genre_id": st.session_state.get("genre_select", "")})),
    )

    new_genre = st.text_input(
        "分野を追加",
        placeholder="新しい分野を入力してください",
        key="new_genre_input",
    )
    if new_genre:
        if new_genre not in genre_id_to_text.values():
            st.subheader(f"{new_genre}の選択肢を設定")
            options_text = st.text_area("選択肢を改行で区切って入力", value="\n".join([]))
            new_genre_options = [
                line.strip() for line in options_text.split("\n") if line.strip()
            ]
            if st.button("選択肢を保存"):
                dispatch((ADD_GENRE, {"genre_text": new_genre}))
                dispatch((EDIT_CHOICES, {"choices": new_genre_options}))

with question_tab:
    st.header("質問を選択または追加")
    if state["genre_id"]:
        genre_data = state["data"][state["genre_id"]]

        question_id_to_text = {}
        question_ids = []
        for question_id, question_data in genre_data["questions"].items():
            question_id_to_text[question_id] = question_data["text"]
            question_ids.append(question_id)

        selected_question_id = st.selectbox(
            f"{genre_data['text']} の質問を選択",
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
            f"{genre_data['text']} の質問を追加",
            placeholder="新しい質問を入力してください",
            key="new_question_input",
            on_change=lambda: dispatch(
                (ADD_QUESTION, {"question_text": st.session_state.new_question_input}))
        )
    else:
        st.warning("分野を選択してください")

with case_tab:
    st.header("場合を選択または追加")
    if state["genre_id"]:
        genre_data = state["data"][state["genre_id"]]

        case_id_to_text = {}
        case_ids = []
        for case_id, case_data in genre_data["cases"].items():
            case_id_to_text[case_id] = case_data["text"]
            case_ids.append(case_id)

        selected_case_id = st.selectbox(
            f"{genre_data['text']} の場合を選択",
            options=case_ids,  # None を先頭に追加
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
            f"{genre_data['text']} の場合を追加",
            placeholder="新しい場合を入力してください",
            key="new_case_input",
            on_change=lambda: dispatch(
                (ADD_CASE, {"case_text": st.session_state.new_case_input}))
        )

    else:
        st.warning("分野を選択してください")

with answer_tab:
    st.header("回答を追加")
    if not state["genre_id"]:
        st.warning("分野を選択してください")
    if not state["question_id"]:
        st.warning("質問を選択してください")
    if not state["case_id"]:
        st.warning("場合を選択してください")
    if state["genre_id"] and state["question_id"] and state["case_id"]:
        genre_data = state["data"][state["genre_id"]]
        st.write(f"分野: **{genre_data['text']}**")
        st.write(
            f"質問: **{genre_data['questions'][state['question_id']]['text']}**")
        st.write(f"場合: **{genre_data['cases'][state['case_id']]['text']}**")
        answer = st.radio(
            "回答を選択", options=genre_data["choices"], horizontal=True)
        if st.button("回答を追加"):
            dispatch((SELECT_CHOICE, {"choice": answer}))
            st.success(
                f"分野: {genre_data['text']}, 場合: {genre_data['cases'][state['case_id']]['text']}, 質問: {genre_data['questions'][state['question_id']]['text']} に対して「{answer}」と回答しました")

st.write("---")
st.write("デバッグ情報")
st.write(state)
