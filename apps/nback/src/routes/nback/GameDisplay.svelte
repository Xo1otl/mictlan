<script lang="ts">
    import * as nback from "../../nback/index";
    import type { Config } from "./ConfigModal.svelte";
    import TrialCard from "./TrialCard.svelte";

    const {
        config,
        onEnd,
    }: {
        config: Config;
        onEnd: (results: nback.TrialResult[]) => void;
    } = $props();

    const trialFactoryOptions = $derived(
        config.trialFactoryOptions ?? nback.DefaultTrialFactoryOptions,
    );

    const [gridRows, gridCols] = $derived(
        trialFactoryOptions.gridSize ?? [1, 1],
    );

    const stimulusTypes: nback.StimulusType[] = $derived(
        trialFactoryOptions.stimulusTypes ?? [],
    );

    let inputs = $derived(
        stimulusTypes.reduce(
            (acc, type) => {
                acc[type] = false;
                return acc;
            },
            {} as Record<nback.StimulusType, boolean>,
        ),
    );

    const toggleInput = (type: nback.StimulusType) => {
        inputs[type] = !inputs[type];
    };

    const readTrialInput: () => nback.MatchResult[] = () => {
        const result: nback.MatchResult[] = [];
        for (const type of stimulusTypes) {
            result.push({
                stimulusType: type,
                match: inputs[type],
            });
        }
        return result;
    };

    let trialCards: Record<string, nback.Trial> = $state({});

    const onUpdate = (
        newTrial: nback.Trial,
        prevTrialResult?: nback.TrialResult,
    ) => {
        trialCards = {};
        addTrialCard(newTrial);
    };

    const onStop = () => {
        onEnd([]);
    };

    const trialFactory = $derived(nback.newTrialFactory(trialFactoryOptions));
    const engine = $derived(
        nback.newTaskEngine({
            ...config.taskEngineOptions,
            trialFactory,
        }),
    );

    // FIXME: 表示したカードは一定時間後消す必要がある
    // FIXME: なぜかカードが同じ位置に来る時に新しいのが表示されない
    const addTrialCard = (trial: nback.Trial) => {
        const [x, y] = trial.stimuli().position ?? [0, 0];
        if (x < 0 || x >= gridCols || y < 0 || y >= gridRows) {
            console.error("Invalid trial position:", trial.stimuli().position);
            return;
        }
        trialCards[`${y}-${x}`] = trial;
    };

    const startTask = () => {
        engine.start(readTrialInput, onUpdate, onStop);
    };

    // FIXME: このロジックマジでどうにかしたい
    const getKey = (index: number): string => {
        const row = Math.floor(index / gridCols);
        const col = index % gridCols;
        return `${row}-${col}`;
    };
</script>

<main class="p-4">
    <button
        type="button"
        onclick={startTask}
        class="px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-600 transition-colors"
    >
        開始
    </button>

    <div
        class="grid gap-2 mt-5"
        style={`grid-template-columns: repeat(${gridCols}, 1fr);`}
    >
        {#each Array(gridRows * gridCols) as _, index}
            <div
                class="flex items-center justify-center border border-gray-300 p-5 aspect-square"
            >
                {#if trialCards[getKey(index)]}
                    <TrialCard trial={trialCards[getKey(index)]} />
                {/if}
            </div>
        {/each}
    </div>

    <div class="mt-5">
        {#each stimulusTypes as type}
            <button
                type="button"
                onclick={() => toggleInput(type)}
                class="mr-2 p-2 border rounded transition-colors"
                class:bg-blue-100={inputs[type]}
                class:bg-white={!inputs[type]}
            >
                {type}
            </button>
        {/each}
    </div>
</main>
