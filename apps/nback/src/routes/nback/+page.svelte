<script lang="ts">
    import { onDestroy } from "svelte";
    import * as nback from "../../nback";
    import ResultList from "./ResultList.svelte";
    import TrialList from "./TrialList.svelte";

    let gameStarted = false;
    let generatedTrials: nback.TrialStimuli[] = [];
    let trialResults: nback.TrialResult[] = [];
    let currentStimuli: nback.TrialStimuli | null = null;
    let engineReset: (() => void) | null = null;

    let characterMatchInput = false;
    let positionMatchInput = false;

    const trialFactory = nback.newTrialFactory({
        stimulusTypes: ["character", "position"],
    });

    const engine = nback.newTaskEngine(2, 20, 2500, trialFactory);

    const readTrialInput = (): nback.MatchResult[] => {
        const results: nback.MatchResult[] = [
            { stimulusType: "character", match: characterMatchInput },
            { stimulusType: "position", match: positionMatchInput },
        ];
        characterMatchInput = false;
        positionMatchInput = false;
        return results;
    };

    const onUpdate = (
        newTrial: nback.Trial,
        prevTrialResult?: nback.TrialResult,
    ): void => {
        if (prevTrialResult) {
            trialResults = [...trialResults, prevTrialResult];
        }
        const stimuli = newTrial.stimuli();
        currentStimuli = stimuli;
        generatedTrials = [...generatedTrials, stimuli];
    };

    const startGame = () => {
        generatedTrials = [];
        trialResults = [];
        currentStimuli = null;
        gameStarted = true;
        engineReset = engine.start(readTrialInput, onUpdate);
    };

    const resetGame = () => {
        if (engineReset) engineReset();
        gameStarted = false;
        generatedTrials = [];
        trialResults = [];
        currentStimuli = null;
    };

    onDestroy(() => {
        if (engineReset) engineReset();
    });

    const onCharacterMatchClick = () => {
        characterMatchInput = true;
    };
    const onPositionMatchClick = () => {
        positionMatchInput = true;
    };

    let showTrials = false;
    let showResults = false;
    const toggleTrials = () => {
        showTrials = !showTrials;
    };
    const toggleResults = () => {
        showResults = !showResults;
    };
</script>

<main>
    <h1>N-back ゲーム</h1>

    {#if !gameStarted}
        <button on:click={startGame}>ゲーム開始</button>
    {:else}
        <button on:click={resetGame}>ゲームリセット</button>
    {/if}

    <section class="grid-container">
        <div class="grid">
            {#each Array(3) as _, rowIndex}
                {#each Array(3) as _, colIndex}
                    <div class="cell">
                        {#if currentStimuli && currentStimuli.position[0] === rowIndex && currentStimuli.position[1] === colIndex}
                            {#if currentStimuli.character}
                                {currentStimuli.character}
                            {:else}
                                ●
                            {/if}
                        {/if}
                    </div>
                {/each}
            {/each}
        </div>
    </section>

    {#if gameStarted}
        <section class="input-section">
            <div class="button-row">
                <button
                    on:click={onCharacterMatchClick}
                    class:active={characterMatchInput}
                >
                    文字マッチ
                </button>
                <button
                    on:click={onPositionMatchClick}
                    class:active={positionMatchInput}
                >
                    位置マッチ
                </button>
            </div>
        </section>
    {/if}

    <section class="toggle-section">
        <div class="button-row">
            <button on:click={toggleTrials}>
                {#if showTrials}履歴を隠す{:else}履歴を表示する{/if}
            </button>
            <button on:click={toggleResults}>
                {#if showResults}結果を隠す{:else}結果を表示する{/if}
            </button>
        </div>
    </section>

    {#if showTrials}
        <TrialList trials={generatedTrials} />
    {/if}

    {#if showResults}
        <ResultList results={trialResults} />
    {/if}
</main>

<style>
    :root {
        --grid-width: min(100vw, 40vh, 380px);
        --grid-gap: 0.2rem;
        --grid-border-color: #ccc;
        --grid-background-color: #f9f9f9;
        --grid-font-size: 2rem;
        --active-button-bg: #ddd;
    }

    main {
        font-family: sans-serif;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }

    button {
        margin: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1rem;
        cursor: pointer;
    }

    button.active {
        background-color: var(--active-button-bg);
    }

    .grid-container {
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1rem;
    }

    .grid {
        max-width: var(--grid-width);
        width: 100%;
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: var(--grid-gap);
        margin: 0 auto;
    }

    .cell {
        aspect-ratio: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--grid-border-color);
        background-color: var(--grid-background-color);
        font-size: var(--grid-font-size);
    }

    .button-row {
        display: flex;
        flex-direction: row;
        gap: 0.5rem;
        justify-content: center;
        align-items: center;
    }
</style>
