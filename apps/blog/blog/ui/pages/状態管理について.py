import streamlit as st

st.set_page_config(layout="wide")

"""
# 状態管理について

## ステートマシン

```typescript
type SimpleState = string;
type RegionName = string;

type OrthognalState = {
	[region: RegionName]: SimpleState | OrthognalState;
};

export type State = SimpleState | OrthognalState;
export type Event = string;

export interface Machine<T extends State, U extends Event> {
	/**
	 * 状態変更の観察者（オブザーバー）を登録する
	 * @param listener 状態遷移が完了し、新しい状態に入った際に呼び出されるコールバック関数
	 * @returns 登録解除のためのメソッドを含むSubscriptionオブジェクト
	 */
	subscribe(listener: (state: T) => void): () => void;

	/**
	 * イベントをディスパッチする。オプションでエフェクトを実行する。
	 * Run-to-Completion (RTC) セマンティクスに従い、エフェクトが提供された場合、
	 * その実行が完了するまで次のイベントは処理されない
	 * @param event ディスパッチするイベント
	 * @param effect オプショナル。イベントに関連付けられたエフェクト（遷移のアクション）
	 * @throws {Error} イベントプールに存在しない、または現在の状態で処理できないイベントを受け取った場合
	 * @returns エフェクトが提供された場合はその実行結果、そうでない場合はundefined
	 */
	dispatch<T>(event: U, effect?: () => T): T | undefined;
}
```

このようなインターフェース考えてみたら割と便利だった

例えばsubscribeにreactのsetStateを渡せば、任意のオブジェクトでデータバインディングできる

状態遷移の実装はなんでもよい。xstateとか使ってもいいし、switch文で頑張って書いてもよい

dispatchにeffectをもたせてエラーのときに状態遷移しないことで、正常な動作しかできない完全な状態遷移を実現できる

## Reducer Pattern

複数の入力欄があるフォームとか考えた時

入力欄Aが有効or無効、入力欄Bが有効or無効、入力欄Cが有効or無効などそれぞれの状態はシンプルでも、フォーム全体としてみれば8通りの複雑な状態がある

ステートマシンでこういうのを管理するのは向いてない、Reducer Patternを使うといい

Reducer Patternはドシンプルで、状態を持つクラスを作って、状態を変更する関数を作って、それを呼び出して一元管理するだけ

一応本家Reactのやつはimmutableとかよくわからんことをやってるけど、普通にstreamlitでmutableにやってみても便利だった
"""
