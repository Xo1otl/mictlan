<script lang="ts">
    import { confetti } from "@neoconfetti/svelte";
    import { onMount } from "svelte";
    import * as nback from "../../nback/index";
    import type { Config } from "./+page";
    import ConfigModal from "./ConfigModal.svelte";
    import GameModal, { type TaskResult } from "./GameModal.svelte";
    import ResultDisplay from "./ResultDisplay.svelte";

    const completeProblemCount = 30;
    let showConfigModal = $state(false);
    let showGameModal = $state(false);

    let config: Config = $state({
        trialFactoryOptions: nback.DefaultTrialFactoryOptions,
        taskEngineOptions: {
            n: 2,
            problemCount: completeProblemCount,
            interval: 4000,
        },
        answerDisplayTime: 600,
    });

    // コンポーネントがマウントされたときに localStorage から config を取得
    onMount(() => {
        const configString = localStorage.getItem("config");
        if (!configString) {
            localStorage.setItem("config", JSON.stringify(config));
            return;
        }
        const savedConfig = JSON.parse(configString);
        if (isValidConfig(savedConfig)) {
            config = savedConfig;
        } else {
            console.error("Invalid config found in localStorage");
            localStorage.setItem("config", JSON.stringify(config));
        }
    });

    // 各フィールドの型検証用関数
    // Config の各フィールドを nback の enum を使って検証する関数
    function isValidConfig(obj: Config): obj is Config {
        if (typeof obj !== "object" || obj === null) return false;

        // answerDisplayTime: number であること
        if (typeof obj.answerDisplayTime !== "number") return false;

        // trialFactoryOptions の検証
        if (
            typeof obj.trialFactoryOptions !== "object" ||
            obj.trialFactoryOptions === null
        )
            return false;
        const tfo = obj.trialFactoryOptions;

        // 各 enum 配列のチェック（存在する場合のみ）
        if ("stimulusTypes" in tfo) {
            if (
                !Array.isArray(tfo.stimulusTypes) ||
                !tfo.stimulusTypes.every((item: nback.StimulusType) =>
                    Object.values(nback.StimulusType).includes(item),
                )
            ) {
                return false;
            }
        }
        if ("colors" in tfo) {
            if (
                !Array.isArray(tfo.colors) ||
                !tfo.colors.every((item: nback.Color) =>
                    Object.values(nback.Color).includes(item),
                )
            ) {
                return false;
            }
        }
        if ("shapes" in tfo) {
            if (
                !Array.isArray(tfo.shapes) ||
                !tfo.shapes.every((item: nback.Shape) =>
                    Object.values(nback.Shape).includes(item),
                )
            ) {
                return false;
            }
        }
        if ("characters" in tfo) {
            if (
                !Array.isArray(tfo.characters) ||
                !tfo.characters.every((item: nback.Character) =>
                    Object.values(nback.Character).includes(item),
                )
            ) {
                return false;
            }
        }
        if ("sounds" in tfo) {
            if (
                !Array.isArray(tfo.sounds) ||
                !tfo.sounds.every((item: nback.Sound) =>
                    Object.values(nback.Sound).includes(item),
                )
            ) {
                return false;
            }
        }
        if ("animations" in tfo) {
            if (
                !Array.isArray(tfo.animations) ||
                !tfo.animations.every((item: nback.Animation) =>
                    Object.values(nback.Animation).includes(item),
                )
            ) {
                return false;
            }
        }

        // gridSize の検証（存在する場合は [number, number] であること）
        if ("gridSize" in tfo) {
            if (
                !Array.isArray(tfo.gridSize) ||
                tfo.gridSize.length !== 2 ||
                typeof tfo.gridSize[0] !== "number" ||
                typeof tfo.gridSize[1] !== "number"
            ) {
                return false;
            }
        }

        // taskEngineOptions の検証（trialFactory は除外）
        if (
            typeof obj.taskEngineOptions !== "object" ||
            obj.taskEngineOptions === null
        )
            return false;
        const teo = obj.taskEngineOptions;
        if (
            typeof teo.n !== "number" ||
            typeof teo.problemCount !== "number" ||
            typeof teo.interval !== "number"
        ) {
            return false;
        }

        return true;
    }

    let taskResult: TaskResult | undefined = $state(undefined);

    const updateResult = async (result: TaskResult) => {
        showGameModal = false;
        taskResult = result;

        if (result.trialResults.length < completeProblemCount) {
            return;
        }

        let n = Object.keys(result.cohensKappa).length;
        const averageKappa =
            Object.values(result.cohensKappa).reduce((sum, value) => {
                if (Number.isNaN(value)) {
                    n--;
                    return sum;
                }
                return sum + value;
            }, 0) / n;
        console.log(averageKappa);
        if (averageKappa > 0.8) {
            taskCompleted = true;
        }
    };

    let taskCompleted = $state(false);

    // config の変更を検知して localStorage に自動保存
    $effect(() => {
        localStorage.setItem("config", JSON.stringify(config));
    });

    const closeModal = () => {
        showConfigModal = false;
    };

    const updateConfig = (newConfig: Config) => {
        config = newConfig;
        closeModal();
    };

    const startTask = () => {
        showGameModal = true;
        taskCompleted = false;
    };
</script>

<main class="p-4 text-column">
    <div class="flex flex-col items-center space-y-4">
        <button
            onclick={() => {
                showConfigModal = true;
            }}
            class="flex items-center gap-2 text-lg hover:underline focus:outline-none"
        >
            <span>Configure task⚙</span>
        </button>
        <button
            type="button"
            onclick={startTask}
            class="relative inline-flex items-center justify-center px-8 py-3 font-semibold tracking-wide text-white transition-all duration-300 ease-out rounded-full bg-gradient-to-r from-green-400 to-blue-500 hover:from-green-500 hover:to-blue-600 shadow-xl transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-300"
        >
            タスク開始
        </button>
    </div>

    {#if showConfigModal}
        <ConfigModal onApply={updateConfig} onCancel={closeModal} {config} />
    {/if}

    {#if showGameModal}
        <GameModal {config} onFinish={updateResult} />
    {/if}

    {#if taskResult}
        <ResultDisplay result={taskResult} {config} />
    {:else}
        <div class="mt-4 space-y-6">
            <section>
                <h2 class="text-2xl font-bold">
                    新たな挑戦で日常をもっと豊かに
                </h2>
                <p>
                    最新の認知科学に基づく N-back task
                    は、タスクの切り替えや注意力を高めるための、シンプルで効果的な脳トレです。短時間で実施できるため、仕事や学習、家事など、さまざまな場面で役立ちます。
                </p>
            </section>

            <section>
                <h3 class="text-xl font-semibold">なぜ N-back task なのか？</h3>
                <p>
                    N-back task
                    は作業記憶を鍛える課題として注目されています。設定は多彩で、音声刺激を有効にすると効果がさらにアップします。また、自分に合った難易度（スコアがギリギリ0.6以上を目指せるレベル）に挑戦することで、より効果的なトレーニングが実現します。
                </p>
            </section>

            <section>
                <h3 class="text-xl font-semibold">スコアの仕組み</h3>
                <p>
                    各刺激に対する反応は Cohen's Kappa
                    によって評価されます。Cohen's Kappa
                    は、期待される反応と実際の反応の一致度を示す指標で、数値が1に近いほど一貫したパフォーマンスを意味します。
                </p>
            </section>

            <section>
                <p class="text-lg font-bold text-green-600">
                    一日10分以上続けよう！
                </p>
                <p>
                    1回30問を5回行えば、10分以上のトレーニングになります。忙しい日々の中でも、継続することで脳の切り替え能力や集中力が確実に向上します。ぜひ、今日から始めてみましょう！
                </p>
            </section>
        </div>
    {/if}
</main>

{#if taskCompleted}
    <div
        style="position: absolute; left: 50%; top: 30%"
        use:confetti={{
            force: 0.7,
            stageWidth: window.innerWidth,
            stageHeight: window.innerHeight,
            colors: ["#ff3e00", "#40b3ff", "#676778"],
        }}
    ></div>
{/if}
