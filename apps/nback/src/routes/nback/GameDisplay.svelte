<script lang="ts">
    import A from "$lib/sounds/A.mp3";
    import B from "$lib/sounds/B.mp3";
    import C from "$lib/sounds/C.mp3";
    import H from "$lib/sounds/H.mp3";
    import K from "$lib/sounds/K.mp3";
    import L from "$lib/sounds/L.mp3";
    import M from "$lib/sounds/M.mp3";
    import O from "$lib/sounds/O.mp3";

    import { blur, fade, fly, scale } from "svelte/transition";
    import * as nback from "../../nback/index";
    import { spin } from "../../transition/transition";
    import type { Config, TaskResult } from "./+page";
    import TrialCard from "./TrialCard.svelte";

    console.log(A);

    let showModal = $state(false);

    const {
        config,
        onEnd,
    }: {
        config: Config;
        onEnd: (results: TaskResult) => void;
    } = $props();

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
    let isDisplayingResult = $state(false);

    // 画面と同期不要な変数
    let nextTrialId = 1;
    let trialResults: nback.TrialResult[] = [];
    let trials: nback.Trial[] = [];

    const getKey = (index: number): string => {
        const row = Math.floor(index / gridCols);
        const col = index % gridCols;
        return `${row}-${col}`;
    };

    const toggleInput = (type: nback.StimulusType) => {
        if (isDisplayingResult) return; // 結果表示中は無視
        inputs[type] = inputs[type] === "selected" ? "none" : "selected";
    };

    const readTrialInput: () => nback.MatchResult[] = () => {
        const result: nback.MatchResult[] = [];
        for (const type of stimulusTypes) {
            result.push({
                stimulusType: type,
                match: inputs[type] === "selected",
            });
        }
        return result;
    };

    const onUpdate = async (
        newTrial?: nback.Trial,
        prevTrialResult?: nback.TrialResult,
    ) => {
        shownTrials = {};

        if (prevTrialResult) {
            console.log("Trial result:", prevTrialResult);
            trialResults.push(prevTrialResult);
            for (const type of stimulusTypes) {
                const r = prevTrialResult.matchResults.find(
                    (m) => m.stimulusType === type,
                );
                inputs[type] = r?.match ? "correct" : "incorrect";
            }
            isDisplayingResult = true;
        }

        lastTrialIndex = prevTrialResult?.trial_idx
            ? prevTrialResult.trial_idx - config.taskEngineOptions.n
            : 0;

        if (newTrial) {
            setTimeout(() => {
                trials.push(newTrial);
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
        onEnd({ trialResults, trials });
        isRunning = false;
        showModal = false;
    };

    const playSound = (sound: nback.Sound) => {
        switch (sound) {
            case nback.Sound.A:
                new Audio(A).play();
                break;
            case nback.Sound.B:
                new Audio(B).play();
                break;
            case nback.Sound.C:
                new Audio(C).play();
                break;
            case nback.Sound.H:
                new Audio(H).play();
                break;
            case nback.Sound.K:
                new Audio(K).play();
                break;
            case nback.Sound.L:
                new Audio(L).play();
                break;
            case nback.Sound.M:
                new Audio(M).play();
                break;
            case nback.Sound.O:
                new Audio(O).play();
                break;
        }
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

    let reset: () => void = () => {};

    const startTask = () => {
        trialResults = [];
        trials = [];
        if (isRunning) {
            console.warn("Task is already running");
            return;
        }
        isRunning = true;
        reset = engine.start(readTrialInput, onUpdate);

        showModal = true;
    };

    const abort = () => {
        isRunning = false;
        reset();
        showModal = false;
        onEnd({ trialResults, trials });
    };
</script>

<main class="p-4">
    <div class="flex justify-center">
        <button
            type="button"
            onclick={startTask}
            class="text-xl text-black underline underline-offset-4"
        >
            タスク開始
        </button>
    </div>

    {#if showModal}
        <div
            class="fixed inset-4 z-50 bg-white/95 backdrop-blur-sm rounded-2xl shadow-xl
                   flex flex-col p-6 border border-gray-200 max-h-screen"
        >
            <p>
                終了: {lastTrialIndex}/{config.taskEngineOptions.problemCount}問
            </p>
            <br />
            <!-- グリッド表示部分 -->
            <div class="flex-grow flex items-center justify-center">
                <div
                    class="grid gap-3 mx-auto w-full h-full max-h-fit max-w-fit"
                    style={`grid-template-columns: repeat(${gridCols}, 1fr);`}
                >
                    {#each Array(gridRows * gridCols) as _, index}
                        <div
                            class="relative flex items-center justify-center border-2
                               border-gray-200 rounded-xl aspect-square overflow-hidden"
                        >
                            {#if shownTrials[getKey(index)]}
                                {#key shownTrials[getKey(index)].trialId}
                                    {#if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Fly}
                                        <div
                                            transition:fly|global={{
                                                y: "100%",
                                            }}
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Scale}
                                        <div
                                            transition:scale|global={{
                                                duration: 1500,
                                                opacity: 100,
                                            }}
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Blur}
                                        <div
                                            transition:blur|global
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.Spin}
                                        <div
                                            in:spin|global={{ duration: 1500 }}
                                            out:fade|global
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {:else if shownTrials[getKey(index)].trial.stimuli().animation === nback.Animation.None}
                                        <div
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {:else}
                                        <div
                                            transition:fade|global
                                            class="absolute inset-0 flex items-center justify-center"
                                        >
                                            <TrialCard
                                                trial={shownTrials[
                                                    getKey(index)
                                                ].trial}
                                            />
                                        </div>
                                    {/if}
                                {/key}
                            {/if}
                        </div>
                    {/each}
                </div>
            </div>

            <!-- 入力ボタン部分 -->
            <div
                class={`grid gap-4 mx-auto mt-4 justify-items-center ${
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
                        class="w-full p-6 rounded-xl border-2 text-xl font-medium transition-all flex items-center justify-center hover:scale-[1.02] active:scale-95"
                        class:border-green-500={inputs[type] === "selected"}
                        class:bg-green-100={inputs[type] === "selected"}
                        class:border-blue-500={inputs[type] === "correct"}
                        class:bg-blue-100={inputs[type] === "correct"}
                        class:border-red-500={inputs[type] === "incorrect"}
                        class:bg-red-100={inputs[type] === "incorrect"}
                        class:border-gray-200={inputs[type] === "none"}
                    >
                        {type}
                    </button>
                {/each}
            </div>

            <div class="mx-auto mt-4 pt-6 border-t border-gray-200">
                <button
                    type="button"
                    onclick={abort}
                    class="py-4 px-6 bg-red-500 hover:bg-red-600 text-white
                           rounded-xl font-medium transition-colors
                           shadow-sm hover:shadow-md"
                >
                    タスクを中断
                </button>
            </div>
        </div>
    {/if}
</main>
