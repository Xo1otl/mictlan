<script lang="ts">
    import {
        DefaultTrialFactoryOptions,
        StimulusType,
        type TrialResult,
    } from "../../nback";
    import type { Config, TaskResult } from "./+page";

    const {
        result = $bindable(),
        config = $bindable(),
    }: { result: TaskResult; config: Config } = $props();

    const {
        trialResultMap,
        totalCount,
        totalCorrect,
        scores,
        overallAccuracy,
    } = $derived.by(() => {
        console.log("calculating derived values");
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

        const scores: Record<string, number> = {};
        const trialFactoryOptions = config.trialFactoryOptions;
        for (const [stimulusType, { correct, total }] of Object.entries(
            typeStats,
        )) {
            // 各 stimulus type に対応する候補配列の長さ（＝候補数）を取得
            let possibleCount: number | undefined;
            switch (stimulusType) {
                case StimulusType.Color:
                    possibleCount = trialFactoryOptions.colors?.length;
                    break;
                case StimulusType.Shape:
                    possibleCount = trialFactoryOptions.shapes?.length;
                    break;
                case StimulusType.Character:
                    possibleCount = trialFactoryOptions.characters?.length;
                    break;
                case StimulusType.Sound:
                    possibleCount = trialFactoryOptions.sounds?.length;
                    break;
                case StimulusType.Animation:
                    possibleCount = trialFactoryOptions.animations?.length;
                    break;
                case StimulusType.Position: {
                    const axis: [number, number] | undefined =
                        trialFactoryOptions.gridSize;
                    possibleCount = axis ? axis[0] * axis[1] : undefined;
                    break;
                }
            }
            if (possibleCount === undefined) {
                throw new Error(
                    `possibleCount is undefined for stimulus type: ${stimulusType}`,
                );
            }
            const expectedAccuracy = 1 - 1 / possibleCount; // 放置時 (常に一致しないを選択) の正答率
            const actualAccuracy = correct / total; // 実際の正答率
            console.log(
                `stimulusType: ${stimulusType}, expectedAccuracy: ${expectedAccuracy}, actualAccuracy: ${actualAccuracy}`,
            );
            // 正規化したスコアを計算： (実際の正答率 - 放置時正答率) / (1 - 放置時正答率)
            const score =
                (actualAccuracy - expectedAccuracy) / (1 - expectedAccuracy);
            scores[stimulusType] = score > 0 ? score : 0;
        }

        const overallAccuracy = ((totalCorrect / totalCount) * 100).toFixed(1);

        return {
            trialResultMap,
            totalCount,
            totalCorrect,
            overallAccuracy,
            scores,
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
        <h2 class="text-xl font-semibold mb-2">StimulusType別スコア</h2>
        <ul class="space-y-2">
            {#each Object.entries(scores) as [type, score]}
                <li class="border p-2 rounded">
                    <strong>{type}</strong>: {score.toFixed(3)}
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
