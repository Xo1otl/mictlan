<script lang="ts">
    import type { TrialResult } from "../../nback";
    import type { TaskResult } from "./+page";

    const { result = $bindable() }: { result: TaskResult } = $props();

    const {
        trialResultMap,
        totalCount,
        totalCorrect,
        typeStats,
        overallAccuracy,
    } = $derived.by(() => {
        const trialResultMap = new Map<number, TrialResult | undefined>();
        for (const trialResult of result.trialResults) {
            trialResultMap.set(trialResult.trial_idx, trialResult);
        }
        const totalCount = result.trialResults.length;

        const typeStats: Record<string, { correct: number; total: number }> =
            {};
        let totalCorrect = 0;
        for (const trialResult of result.trialResults) {
            let isCorrect = true;
            for (const match of trialResult.matchResults) {
                if (!match.match) isCorrect = false;
                if (!typeStats[match.stimulusType]) {
                    typeStats[match.stimulusType] = {
                        correct: 0,
                        total: 0,
                    };
                }
                typeStats[match.stimulusType].total++;
                if (match.match) typeStats[match.stimulusType].correct++;
            }
            if (isCorrect) totalCorrect++;
        }
        const overallAccuracy = ((totalCorrect / totalCount) * 100).toFixed(1);

        return {
            trialResultMap,
            totalCount,
            totalCorrect,
            overallAccuracy,
            typeStats,
        };
    });
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
