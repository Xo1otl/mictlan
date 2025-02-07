<script lang="ts">
    import { onMount } from "svelte";
    import * as nback from "../../nback/index";
    import type { Config } from "./+page";
    import ConfigModal from "./ConfigModal.svelte";
    import DebugCard from "./DebugCard.svelte";
    import GameDisplay, { type TaskResult } from "./GameDisplay.svelte";
    import ResultDisplay from "./ResultDisplay.svelte";

    let showModal = $state(false);

    // localStorage に設定があればそれを使い、なければデフォルトを使用
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
        const savedConfig = localStorage.getItem("config");
        if (savedConfig) {
            config = JSON.parse(savedConfig);
        }
    });

    let taskResult: TaskResult | undefined = $state(undefined);

    const onEnd = (result: TaskResult) => {
        taskResult = result;
    };

    // config の変更を検知して localStorage に自動保存
    $effect(() => {
        localStorage.setItem("config", JSON.stringify(config));
    });
</script>

<main class="p-4 text-column">
    <div class="flex justify-center">
        <button
            onclick={() => {
                showModal = true;
            }}
            class="flex items-center gap-2 text-lg hover:underline focus:outline-none"
        >
            <span>Configure task⚙</span>
        </button>
    </div>

    {#if showModal}
        <ConfigModal bind:showModal bind:config />
    {/if}

    <GameDisplay {config} {onEnd} />

    {#if taskResult}
        <ResultDisplay bind:result={taskResult} />
    {:else}
        <!-- 日常生活に直結するメリットを強調した N-back task の紹介 -->
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
