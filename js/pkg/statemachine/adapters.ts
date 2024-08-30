export interface InterfaceAdapter<State extends string, Event extends string> {
	/**
	 * コールバック処理を渡すことができる状態遷移
	 * 処理に成功すると事後状態になる
	 * @param event 処理の事前イベント
	 * @param action イベントと対応した処理
	 * @throws {Error} 未定義のイベントを受け取った時
	 * @returns イベントに対応する処理の結果
	 */
	transition<T>(event: Event, action: () => T): T;

	/**
	 * 状態の変更を監視するリスナーを登録する
	 * @param listener 状態が変更されたときに呼び出されるコールバック関数
	 * @returns 登録したリスナーの購読を解除する関数
	 */
	subscribe(listener: (state: State) => void): () => void;
}
