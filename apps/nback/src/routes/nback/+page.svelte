<script lang="ts">
    import * as nback from "../../nback/index";
    import type { Config } from "./+page";
    import ConfigModal from "./ConfigModal.svelte";
    import DebugCard from "./DebugCard.svelte";
    import GameDisplay from "./GameDisplay.svelte";
    import ResultDisplay from "./ResultDisplay.svelte";

    let showModal = $state(false);
    let config: Config = $state({
        trialFactoryOptions: nback.DefaultTrialFactoryOptions,
        taskEngineOptions: {
            n: 2,
            problemCount: 20,
            interval: 3600,
        },
        answerDisplayTime: 600,
    });

    let results: nback.TrialResult[] = $state([]);

    const onEnd = (r: nback.TrialResult[]) => {
        results = r;
    };
</script>

<main class="p-4">
    <button
        onclick={() => {
            showModal = true;
        }}
        class="px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-600 transition-colors"
    >
        設定を変更
    </button>

    {#if showModal}
        <ConfigModal bind:showModal bind:config />
    {/if}

    <GameDisplay {config} {onEnd} />
    <DebugCard value={config} />
    {#if results.length > 0}
        <ResultDisplay {results} />
    {/if}
</main>
