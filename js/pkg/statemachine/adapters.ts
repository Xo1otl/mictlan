type OrthognalState = {
	[region: string]: string | OrthognalState;
};

export type StateBase = string | OrthognalState

export interface InterfaceAdapter<
	State extends StateBase,
	Event extends string,
> {
	/**
	 * イベントをディスパッチし、関連するエフェクトを実行する
	 * Run-to-Completion (RTC) セマンティクスに従い、エフェクトの実行が完了するまで次のイベントは処理されない
	 * @param event ディスパッチするイベント
	 * @param effect イベントに関連付けられたエフェクト（遷移のアクション）
	 * @throws {Error} イベントプールに存在しない、または現在の状態で処理できないイベントを受け取った場合
	 * @returns エフェクトの実行結果
	 */
	dispatch<T>(event: Event, effect: () => T): T;

	/**
	 * 状態変更の観察者（オブザーバー）を登録する
	 * @param listener 状態遷移が完了し、新しい状態に入った際に呼び出されるコールバック関数
	 * @returns 登録した観察者の購読を解除する関数
	 */
	subscribe(listener: (state: State) => void): () => void;
}
