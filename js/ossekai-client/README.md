# ossekai-client

authApp は stateMachine で state 管理をする
authApp はただのオブジェクトなので react で使用できるように useStateMachine で wrap する
provider や hook や context などで authApp を保持し、コンポーネントで authApp の状態に応じた処理を行う

# Note

useMemo は関数を実行せず戻り値を再利用したい場合だけ、それ以外は useCallback でいい

- state actor について

状態遷移図は、主語をこのアプリケーションとして、事前イベントと事後状態で書く

state actor の初期状態について、session の値が実体なので、state actor の snapshot を保存するのではなく、初期状態は必ず session の値から導出する

# TODO

画面から auth を呼び出す時は必ず同期が取れるがロジック上で state が変わった時に画面がそれについていかない問題がある

authApp の state を subscribe できるようにして、そこで setState を呼ぶようにすれば画面と同期が取れる
