<script lang="ts">
    import { onDestroy } from "svelte";
    import * as nback from "../../nback";

    // ゲーム状態や生成されたトライアルを管理するリアクティブ変数
    let gameStarted = false;
    let generatedTrials: nback.TrialStimuli[] = [];
    // エンジンのリセット（停止）関数を保持する変数（エンジン実装によっては利用可能）
    let engineReset: (() => void) | null = null;

    // nback ライブラリの trialFactory を生成（使用する刺激の種類を指定）
    const trialFactory = nback.newTrialFactory({
        stimulusTypes: ["character", "position"],
    });

    // 2-back、20トライアル、各トライアルの間隔を2000msに設定したタスクエンジンを生成
    const engine = nback.newTaskEngine(2, 20, 2000, trialFactory);

    /**
     * readTrialInput は各トライアルごとにユーザー入力を読み取る関数です。
     * ここではダミー実装として常に false を返しています。
     * 実際のゲームでは、ボタン押下やキーボード入力に応じた結果を返すようにしてください。
     */
    const readTrialInput = (): nback.MatchResult[] => {
        return [
            { stimulusType: "character", match: false },
            { stimulusType: "position", match: false },
        ];
    };

    /**
     * onGenerateTrial はエンジンから新しいトライアルが生成されるたびに呼ばれるコールバックです。
     * この例では、生成された刺激を generatedTrials 配列に追加して UI に表示します。
     */
    const onGenerateTrial = (trial: nback.Trial): void => {
        // Svelte の再描画を促すためにスプレッド構文で新しい配列をセット
        generatedTrials = [...generatedTrials, trial.stimuli()];
    };

    /**
     * onStop は全トライアル終了後に呼ばれるコールバックです。
     * 結果をコンソールに表示し、ゲーム状態を終了に変更します。
     */
    const onStop = (results: nback.TrialResult[]) => {
        console.log("ゲーム終了。結果:", results);
        gameStarted = false;
    };

    /**
     * startGame はゲーム開始時に呼ばれる関数です。
     * 内部状態のリセット後、エンジンをスタートします。
     */
    const startGame = () => {
        // UI上のトライアル一覧をリセット
        generatedTrials = [];
        gameStarted = true;
        // エンジン開始。戻り値としてリセット用の関数（もし提供されていれば）を保持
        engineReset = engine.start(readTrialInput, onGenerateTrial, onStop);
    };

    /**
     * resetGame はゲーム中断・リセット用の関数です。
     * エンジンの停止処理（提供されている場合）を呼び出し、状態を初期化します。
     */
    const resetGame = () => {
        if (engineReset) {
            engineReset();
        }
        gameStarted = false;
        generatedTrials = [];
    };

    // コンポーネントが破棄される際に、エンジンの停止処理を実行（メモリリーク防止用）
    onDestroy(() => {
        if (engineReset) {
            engineReset();
        }
    });
</script>

<main>
    <h1>N-back タスク ゲーム</h1>

    {#if !gameStarted}
        <!-- ゲーム未開始時は開始ボタンを表示 -->
        <button on:click={startGame}>ゲーム開始</button>
    {:else}
        <!-- ゲーム開始中はリセットボタンを表示 -->
        <button on:click={resetGame}>ゲームリセット</button>
    {/if}

    <section>
        <h2>生成されたトライアル</h2>
        {#if generatedTrials.length === 0}
            <p>まだトライアルは生成されていません。</p>
        {:else}
            <ul>
                {#each generatedTrials as trial, index}
                    <li>
                        <strong>トライアル {index + 1}:</strong>
                        {#if trial.character}
                            <span> キャラクター: {trial.character}</span>
                        {/if}
                        {#if trial.position}
                            <span>
                                - 位置: ({trial.position[0]}, {trial
                                    .position[1]})</span
                            >
                        {/if}
                    </li>
                {/each}
            </ul>
        {/if}
    </section>
</main>

<style>
    main {
        font-family: sans-serif;
        padding: 1rem;
    }
    button {
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
    }
    ul {
        list-style: none;
        padding: 0;
    }
    li {
        margin: 0.5rem 0;
        padding: 0.25rem;
        background-color: #f4f4f4;
        border-radius: 4px;
    }
</style>
