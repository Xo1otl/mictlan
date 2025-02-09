<script lang="ts">
    import type { TrialResult } from "../../nback";
    import type { Config } from "./+page";
    import type { TaskResult } from "./GameModal.svelte";

    const { result, config }: { result: TaskResult; config: Config } = $props();

    const {
        trialResultMap,
        totalCount,
        totalCorrect,
        cohensKappa,
        overallAccuracy,
        shareButtonKey,
    } = $derived.by(() => {
        console.log("calculating derived values");
        const trialResultMap = new Map<number, TrialResult>();
        for (const trialResult of result.trialResults) {
            trialResultMap.set(trialResult.trialIdx, trialResult);
        }
        const totalCount = result.trialResults.length;

        let totalCorrect = 0;
        for (const trialResult of result.trialResults) {
            let isCorrect = true;
            for (const match of trialResult.matchResults) {
                if (!match.match) isCorrect = false;
            }
            if (isCorrect) totalCorrect++;
        }

        const overallAccuracy = ((totalCorrect / totalCount) * 100).toFixed(1);

        const shareButtonKey = crypto.randomUUID();

        return {
            trialResultMap,
            totalCount,
            totalCorrect,
            overallAccuracy,
            cohensKappa: result.cohensKappa,
            shareButtonKey,
        };
    });

    let twitterWidgetContainer: HTMLDivElement | null = null;
    $effect(() => {
        if (!twitterWidgetContainer) return;
        for (const child of Array.from(twitterWidgetContainer.children)) {
            if (child.id === shareButtonKey) continue;
            twitterWidgetContainer.removeChild(child);
            // ここで何も削除されない場合、コンテンツブロッカーによってツイートボタンが表示されていない可能性がある
        }
        window.twttr.widgets.load(twitterWidgetContainer);
    });
</script>

<div class="p-4 space-y-8">
    <div bind:this={twitterWidgetContainer}>
        {#key shareButtonKey}
            <a
                id={shareButtonKey}
                href="https://twitter.com/share?ref_src=twsrc%5Etfw"
                class="twitter-share-button"
                data-text={`私の${config.taskEngineOptions.n} Backタスクの結果は: ${totalCorrect}/${totalCount} (正答率: ${overallAccuracy}%) | タイプ別スコア: ${config.trialFactoryOptions.stimulusTypes?.map((type) => `${type}:${cohensKappa[type].toFixed(3)}`).join(" | ")} でした。\n君も挑戦して、一緒にハイスコアを目指しましょう！！`}
                data-hashtags="nback"
                data-show-count="false">Tweet</a
            >
        {/key}
    </div>
    <!-- 全体集計 -->
    <div>
        <h1 class="text-2xl font-bold mb-2">
            N={config.taskEngineOptions.n} タスク結果
        </h1>
        <p>正答数: {totalCorrect} / {totalCount}</p>
        <p>正答率: {overallAccuracy}%</p>
    </div>

    <!-- StimulusType別集計 -->
    <div>
        <h2 class="text-xl font-semibold mb-2">StimulusType別スコア</h2>
        <ul class="space-y-2">
            {#each config.trialFactoryOptions.stimulusTypes! as type}
                <li class="border p-2 rounded">
                    <strong>{type}</strong>: {cohensKappa[type].toFixed(3)}
                </li>
            {/each}
        </ul>
    </div>

    <!-- 各トライアル詳細（trialごとにループ） -->
    <div>
        <h2 class="text-xl font-semibold mb-2">各トライアル詳細</h2>
        <ul class="space-y-4">
            {#each result.trials as trial, index}
                <li class="border p-3 rounded">
                    <!-- trial.index は 0 始まりなので +1 して表示 -->
                    <h3 class="font-bold mb-2">トライアル {index + 1}</h3>
                    <!-- 刺激履歴と判定結果を横並びに -->
                    <div
                        class="flex flex-col md:flex-row md:space-x-6 space-y-4 md:space-y-0"
                    >
                        <!-- 刺激履歴 -->
                        <div class="flex-1">
                            <p class="underline mb-1">刺激履歴</p>
                            {#if trial.stimuli}
                                {#each Object.entries(trial.stimuli()) as [stimulusType, value]}
                                    <p class="whitespace-nowrap">
                                        <span class="font-medium"
                                            >{stimulusType}:</span
                                        >
                                        {JSON.stringify(value)}
                                    </p>
                                {/each}
                            {:else}
                                <p>刺激データなし</p>
                            {/if}
                        </div>
                        <!-- 判定結果 -->
                        <div class="flex-1">
                            <p class="underline mb-1">判定結果</p>
                            {#if trialResultMap.get(index + 1)}
                                {@const trialResult = trialResultMap.get(
                                    index + 1,
                                )}
                                <ul class="list-disc pl-5">
                                    {#each trialResult!.matchResults as match}
                                        <li>
                                            {match.stimulusType}: {match.match
                                                ? "○"
                                                : "×"}
                                        </li>
                                    {/each}
                                </ul>
                            {:else}
                                <p>判定データなし</p>
                            {/if}
                        </div>
                    </div>
                </li>
            {/each}
        </ul>
    </div>
</div>
