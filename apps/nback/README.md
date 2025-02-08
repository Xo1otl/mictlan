# nback

## TODO

* [pwaのテンプレート](https://github.com/vite-pwa/sveltekit/tree/main/examples/sveltekit-ts)
* [pwa](https://github.com/vite-pwa/sveltekit/tree/main?tab=readme-ov-file)
    
## FIXME

* useReducerとdispatchの組み合わせみたいなんができたらいいかもしれない
    * reducerがstate machineみたいな感じ
    * dispatchがstate machineに対してeventを送信する感覚

## domain

### Mutation

#### Task設定保存
* **TaskSettings**
    * Stimulusとスタックの長さとinterval
    * Stimulusは自身の(見た目以外の論理的な)情報をカプセル化する
        * coordinates: max_x, max_y
        * color: colors
        * shape: shapes
        * character: available characters

#### ゲームプレイ履歴保存
* **TaskInfo**
    * trial_index、設定、プレイした日付のセット
* **TrialData**
    * task_id
    * trial_idx
    * Stimulus
    * 正or誤

### Query

#### Task設定取得
* **TaskSettings**
    * Stimulusとスタックの長さ

#### ゲームプレイ履歴取得
* **TaskInfo**
    * 有効化されているStimulus
    * スタックの長さ
* **TrialData**
    * (trial_idx,stimulus) 毎の正答率
    * trial_idxごとの正答率 (計算できる)
    * Stimulus毎の正答率 (計算できる)

## ui

`A -> B -> C -> D`というTrialにおいて、n=2の時、Cが来た時にAと比較する

### TaskEngine 
Trialの生成をsubscribeできるようにする
入力をする関数を用意

1. 設定からStimulus情報とnとintervalを取得 
2. ステートを初期化
    * スタックの長さはn+1
3. 一定intervalで以下の処理を行うバックグラウンドタスクを行う
    * スタックの末尾にTrialが存在する場合、入力を読み、末尾のTrialと比較、ステートの正誤結果に追記、入力を空にする
    * Stimulusの値をランダムに決めてTrialを生成、listenerを呼び出す
    * 表示したStimulusをスタックに追加、スタックが一杯の場合、末尾のTrialを削除

### TaskState
* 現在のTrial番号
* TrialをStackに積んで保持
    * Trial: 有効化されているStimulusをまとめたもの
    * Stack: n+1個のオブジェクトを積んでおくスタック
* 正誤結果
    * trial_idx、Stimulus、 正誤
    * {"trial_idx": 1, "stimulus": "color", isCorrect: False}
