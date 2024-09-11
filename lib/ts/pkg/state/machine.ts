type SimpleState = string;

type OrthognalState = {
	[region: SimpleState]: SimpleState | OrthognalState;
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
