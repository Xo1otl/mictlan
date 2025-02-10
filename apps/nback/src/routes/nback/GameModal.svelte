<script module lang="ts">
    export type TaskResult = {
        cohensKappa: Record<nback.StimulusType, number>;
        trials: nback.Trial[];
        trialResults: nback.TrialResult[];
    };

    const newTaskResult = (): TaskResult => ({
        trialResults: [],
        trials: [],
        cohensKappa: {
            [nback.StimulusType.Position]: Number.NaN,
            [nback.StimulusType.Color]: Number.NaN,
            [nback.StimulusType.Character]: Number.NaN,
            [nback.StimulusType.Shape]: Number.NaN,
            [nback.StimulusType.Sound]: Number.NaN,
            [nback.StimulusType.Animation]: Number.NaN,
        },
    });
</script>

<script lang="ts">
    import A from "$lib/sounds/A.mp3";
    import B from "$lib/sounds/B.mp3";
    import C from "$lib/sounds/C.mp3";
    import H from "$lib/sounds/H.mp3";
    import K from "$lib/sounds/K.mp3";
    import L from "$lib/sounds/L.mp3";
    import M from "$lib/sounds/M.mp3";
    import O from "$lib/sounds/O.mp3";

    import { onMount } from "svelte";
    import { blur, fade, fly, scale } from "svelte/transition";
    import * as nback from "../../nback/index";
    import { spin } from "../../transition/transition";
    import type { Config } from "./+page";
    import TrialCard from "./TrialCard.svelte";

    const {
        config,
        onFinish,
    }: {
        config: Config;
        onFinish: (results: TaskResult) => void;
    } = $props();

    onMount(() => {
        // 画面が表示されてからタスクを開始
        const engineReset = engine.start(provideInput, updateComponent);
        finish = () => {
            engineReset();
            showNextTrialTimer ?? clearTimeout(showNextTrialTimer);
            isRunning = false;
            onFinish(taskResult);
        };
    });

    const provideInput: () => nback.MatchResult[] = () => {
        const result: nback.MatchResult[] = [];
        for (const type of stimulusTypes) {
            result.push({
                stimulusType: type,
                match: inputs[type] === "selected",
            });
        }
        return result;
    };

    const updateComponent = async (
        newTrial?: nback.Trial,
        prevTrialResult?: nback.TrialResult,
    ) => {
        shownTrials = {};

        if (prevTrialResult) {
            console.log("Trial result:", prevTrialResult);
            taskResult.cohensKappa = prevTrialResult.cohensKappa;
            taskResult.trialResults.push(prevTrialResult);
            for (const type of stimulusTypes) {
                const r = prevTrialResult.matchResults.find(
                    (m) => m.stimulusType === type,
                );
                inputs[type] = r?.match ? "correct" : "incorrect";
            }
            isDisplayingResult = true;
        }

        lastTrialIndex = prevTrialResult?.trialIdx
            ? prevTrialResult.trialIdx - config.taskEngineOptions.n
            : 0;

        if (newTrial) {
            showNextTrialTimer = setTimeout(() => {
                taskResult.trials.push(newTrial);
                for (const type of stimulusTypes) {
                    inputs[type] = "none";
                }
                isDisplayingResult = false;
                addTrialEntry(newTrial);
            }, config.answerDisplayTime);
            return;
        }

        console.log("Task が終了しました");
        await new Promise((resolve) => setTimeout(resolve, 1000));
        finish();
    };

    const addTrialEntry = (trial: nback.Trial) => {
        const sound = trial.stimuli().sound;
        if (sound) {
            playSound(sound);
        }
        const [col, row] = trial.stimuli().position ?? [0, 0];
        if (row < 0 || row >= gridRows || col < 0 || col >= gridCols) {
            console.error("Invalid trial position:", trial.stimuli().position);
            return;
        }
        const coordinate = `${row}-${col}`;
        shownTrials[coordinate] = { trial, trialId: nextTrialId++ };
    };

    const abort = () => {
        finish();
    };

    const toggleInput = (type: nback.StimulusType) => {
        if (isDisplayingResult) return; // 結果表示中は無視
        inputs[type] = inputs[type] === "selected" ? "none" : "selected";
    };

    // 最初にロードしないとsafaridで音が鳴らない
    const audioA = new Audio(A);
    const audioB = new Audio(B);
    const audioC = new Audio(C);
    const audioH = new Audio(H);
    const audioK = new Audio(K);
    const audioL = new Audio(L);
    const audioM = new Audio(M);
    const audioO = new Audio(O);

    const playSound = (sound: nback.Sound) => {
        switch (sound) {
            case nback.Sound.A:
                audioA.play();
                break;
            case nback.Sound.B:
                audioB.play();
                break;
            case nback.Sound.C:
                audioC.play();
                break;
            case nback.Sound.H:
                audioH.play();
                break;
            case nback.Sound.K:
                audioK.play();
                break;
            case nback.Sound.L:
                audioL.play();
                break;
            case nback.Sound.M:
                audioM.play();
                break;
            case nback.Sound.O:
                audioO.play();
                break;
        }
    };

    // 不変のステート
    const { gridRows, gridCols, stimulusTypes, engine, inputs } = $derived.by(
        () => {
            const trialFactoryOptions =
                config.trialFactoryOptions ?? nback.DefaultTrialFactoryOptions;
            const [gridCols, gridRows] = trialFactoryOptions.gridSize ?? [1, 1];
            const stimulusTypes: nback.StimulusType[] =
                trialFactoryOptions.stimulusTypes ?? [];
            const inputs = $state(
                stimulusTypes.reduce(
                    (acc, type) => {
                        acc[type] = "none";
                        return acc;
                    },
                    {} as Record<
                        nback.StimulusType,
                        "none" | "selected" | "correct" | "incorrect"
                    >,
                ),
            );
            const trialFactory = nback.newTrialFactory(trialFactoryOptions);
            const engine = nback.newTaskEngine({
                ...config.taskEngineOptions,
                trialFactory,
            });
            return {
                gridRows,
                gridCols,
                stimulusTypes,
                engine,
                inputs,
            };
        },
    );

    // 可変のステート
    let shownTrials: Record<string, { trial: nback.Trial; trialId: number }> =
        $state({});
    let isRunning = $state(false);
    let lastTrialIndex = $state(0);
    let isDisplayingResult = $state(false); // 直前のトライアルの結果表示中の場合入力を無視する

    // 画面と同期不要な変数
    let nextTrialId = 1;
    let taskResult: TaskResult = newTaskResult();
    const getKey = (index: number): string => {
        const row = Math.floor(index / gridCols);
        const col = index % gridCols;
        return `${row}-${col}`;
    };
    let showNextTrialTimer: ReturnType<typeof setTimeout>;
    let finish: () => void = () => {};
</script>

<!-- modal -->
<div
    class="fixed inset-4 z-50 bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl p-6 border border-gray-200"
>
    <!-- container -->
    <div
        class="container max-w-2xl mx-auto h-full grid grid-rows-[auto,1fr,auto]"
    >
        <!-- ヘッダー領域 -->
        <div>
            <p>
                終了: {lastTrialIndex}/{config.taskEngineOptions.problemCount}問
            </p>
            <br />
        </div>

        <!-- カードグリッド領域：grid の行が 1fr になっているため、ヘッダーとフッター以外の残り全体を占有 -->
        <div
            id="trialgrid"
            class="grid gap-3 items-center justify-center"
            style="
          grid-template-columns: repeat({gridCols}, minmax(0, 1fr));
          grid-auto-rows: 1fr;
        "
        >
            {#each Array(gridRows * gridCols) as _, index}
                <div
                    class="relative items-center justify-center border-2 border-gray-200 rounded-xl w-full h-full"
                >
                    {#if shownTrials[getKey(index)]}
                        {#key shownTrials[getKey(index)].trialId}
                            {#if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Fly}
                                <div
                                    transition:fly|global={{ y: "100%" }}
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Scale}
                                <div
                                    in:scale|global={{
                                        duration: 1500,
                                        start: 0.1,
                                    }}
                                    out:scale|global={{
                                        duration: 400,
                                        start: 0.1,
                                    }}
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Blur}
                                <div
                                    in:blur|global={{ duration: 1200 }}
                                    out:blur|global={{ duration: 400 }}
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Spin}
                                <div
                                    in:spin|global={{ duration: 1500 }}
                                    out:fade|global
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.None}
                                <div
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {:else}
                                <div
                                    transition:fade|global
                                    class="absolute inset-0 flex items-center justify-center"
                                >
                                    <TrialCard
                                        trial={shownTrials[getKey(index)].trial}
                                    />
                                </div>
                            {/if}
                        {/key}
                    {/if}
                </div>
            {/each}
        </div>

        <!-- フッター領域：入力ボタンとタスク中断ボタン -->
        <div class="space-y-4 mt-4">
            <!-- 入力ボタン部分 -->
            <div
                class={`grid gap-4 mx-auto justify-items-center ${
                    stimulusTypes.length <= 2
                        ? `grid-cols-${stimulusTypes.length}`
                        : stimulusTypes.length <= 4
                          ? "grid-cols-2"
                          : "grid-cols-3"
                }`}
            >
                {#each stimulusTypes as type}
                    <button
                        type="button"
                        onclick={() => toggleInput(type)}
                        class="w-full p-6 rounded-xl border-2 sm:text-xl font-medium transition-all flex items-center justify-center
                hover:scale-[1.02] active:scale-95
                {inputs[type] === 'selected'
                            ? 'border-green-500 bg-green-100'
                            : ''}
                {inputs[type] === 'correct'
                            ? 'border-blue-500 bg-blue-100'
                            : ''}
                {inputs[type] === 'incorrect'
                            ? 'border-red-500 bg-red-100'
                            : ''}
                {inputs[type] === 'none' ? 'border-gray-200' : ''}"
                    >
                        {type}
                    </button>
                {/each}
            </div>

            <!-- タスク中断ボタン -->
            <div
                class="flex justify-center mx-auto pt-6 border-t border-gray-200"
            >
                <button
                    type="button"
                    onclick={abort}
                    class="py-4 px-6 bg-red-500 hover:bg-red-600 text-white rounded-xl font-medium transition-colors shadow-sm hover:shadow-md"
                >
                    タスクを中断
                </button>
            </div>
        </div>
    </div>
</div>
