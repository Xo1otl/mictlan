<script lang="ts">
    import { confetti } from "@neoconfetti/svelte";
    import { onMount } from "svelte";
    import * as nback from "../../nback/index";
    import type { Config } from "./+page";
    import ConfigModal from "./ConfigModal.svelte";
    import GameModal, { type TaskResult } from "./GameModal.svelte";
    import ResultDisplay from "./ResultDisplay.svelte";

    let showConfigModal = $state(false);
    let showGameModal = $state(false);

    let config: Config = $state({
        trialFactoryOptions: nback.DefaultTrialFactoryOptions,
        taskEngineOptions: {
            n: 2,
            problemCount: 20,
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

        if (result.trialResults.length < 20) {
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
        if (averageKappa > 0.75) {
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
                    あなたの日常を変える、新たな挑戦へ
                </h2>
                <p>
                    最新の認知科学に裏打ちされた N-back task
                    は、単なる脳トレではなく、
                    <strong
                        >日々のタスク切り替えや注意力の向上、さらには集中が続きにくいと感じる方にも効果が期待される</strong
                    >
                    実践的なトレーニングプログラムです。多忙な現代生活の中で、情報の切り替えがスムーズになると、仕事や学習、さらには家事やコミュニケーションにも大きなプラス効果をもたらします。
                </p>
            </section>

            <section>
                <h3 class="text-xl font-semibold">なぜ N-back task なのか？</h3>
                <p>
                    N-back task
                    は、作業記憶の維持・更新を鍛えるためのシンプルかつ効果的な課題です。<br
                    />
                    研究によれば、N-back task を含むトレーニングは、日常生活での迅速なタスク切り替え能力や注意力の改善に寄与する可能性が示されています。<br
                    />
                    さらに、一部の研究では、作業記憶トレーニングが、集中力が続きにくいと感じる方や、ADHD
                    の症状に効果が期待されることも報告されており (<a
                        href="https://doi.org/10.3390/brainsci10100715"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="text-blue-600 underline">Dotare et al. (2020)</a
                    >)、現実の生活の質向上に直結することが期待されます。
                </p>
            </section>

            <section>
                <h3 class="text-xl font-semibold">スコアの説明</h3>
                <p>
                    このシステムでは、各刺激ごとの反応を Cohen&apos;s Kappa
                    を用いて評価しています。<br />
                    Cohen&apos;s Kappa は、期待される反応と実際の反応との一致度を示す指標で、偶然の一致を補正した数値です。<br
                    />
                    数値が1に近いほど、反応が一貫しており、設定された基準に沿ったパフォーマンスが発揮されていることを意味します。<br
                    />
                    つまり、このスコアを通じて、あなたのパフォーマンスがどれだけ期待値に近いかを確認できます。
                </p>
            </section>

            <section>
                <p class="text-lg font-bold text-green-600">
                    今こそ、あなたの可能性に挑戦する時です！
                </p>
                <p>
                    どんなに忙しい毎日でも、ほんの少しの時間で脳の切り替え能力や集中力は鍛えられます。<br
                    />
                    科学的根拠に基づいたこの N-back task をぜひ実践してみてください。<br
                    />
                    小さな積み重ねが、あなたの日常をより快適に、そして充実させる鍵となるでしょう！
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
