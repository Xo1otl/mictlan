<script lang="ts">
    import type { TrialResult } from "../../nback";
    import type { TaskResult } from "./+page";

    // Svelte 5 using runes: use $props() instead of `export let`
    const { result }: { result: TaskResult } = $props();

    // trialResults を trial_idx ごとにマップ化（trial_idx は trials の index + 1 と仮定）
    const trialResultMap = new Map<number, TrialResult | undefined>();
    for (const trialResult of result.trialResults) {
        trialResultMap.set(trialResult.trial_idx, trialResult);
    }

    // 全体・StimulusTypeごとの正答数・総数を集計
    let totalCorrect = $state(0);
    let totalCount = $state(0);
    const typeStats: Record<string, { correct: number; total: number }> = {};

    for (const trialResult of result.trialResults) {
        for (const match of trialResult.matchResults) {
            totalCount++;
            if (match.match) totalCorrect++;
            if (!typeStats[match.stimulusType]) {
                typeStats[match.stimulusType] = { correct: 0, total: 0 };
            }
            typeStats[match.stimulusType].total++;
            if (match.match) typeStats[match.stimulusType].correct++;
        }
    }

    const overallAccuracy = $derived(
        totalCount ? ((totalCorrect / totalCount) * 100).toFixed(1) : "0.0",
    );
</script>

<section class="p-4 max-w-3xl mx-auto space-y-8">
    <!-- 全体集計 -->
    <div>
        <h1 class="text-2xl font-bold mb-2">タスク結果</h1>
        <p>正答数: {totalCorrect} / {totalCount}</p>
        <p>正答率: {overallAccuracy}%</p>
    </div>

    <!-- StimulusType別集計 -->
    <div>
        <h2 class="text-xl font-semibold mb-2">StimulusType別正答率</h2>
        <ul class="space-y-2">
            {#each Object.entries(typeStats) as [type, { correct, total }]}
                <li class="border p-2 rounded">
                    <strong>{type}</strong>: {correct} / {total} (
                    {total ? ((correct / total) * 100).toFixed(1) : "0.0"}%)
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
                                    <p>
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
</section>
