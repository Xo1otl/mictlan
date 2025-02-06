<script lang="ts">
    import * as nback from "../../nback/index";
    import type { Config, TaskResult } from "./+page";
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

    let taskResult: TaskResult | undefined = $state(undefined);

    const onEnd = (result: TaskResult) => {
        console.log("Task ended with trialResults", result.trialResults);
        taskResult = result;
    };
</script>

<main class="p-4">
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
    {/if}
</main>
