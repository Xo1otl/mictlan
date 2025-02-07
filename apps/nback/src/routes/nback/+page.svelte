<script lang="ts">
    import { onMount } from "svelte";
    import * as nback from "../../nback/index";
    import type { Config, TaskResult } from "./+page";
    import ConfigModal from "./ConfigModal.svelte";
    import DebugCard from "./DebugCard.svelte";
    import GameDisplay from "./GameDisplay.svelte";
    import ResultDisplay from "./ResultDisplay.svelte";

    let showModal = $state(false);

    // localStorage に設定があればそれを使い、なければデフォルトを使用
    let config: Config = $state({
        trialFactoryOptions: nback.DefaultTrialFactoryOptions,
        taskEngineOptions: {
            n: 2,
            problemCount: 20,
            interval: 4000,
        },
        answerDisplayTime: 600,
    });

    // コンポーネントがマウントされたときに localStorage から config を取得
    onMount(() => {
        const savedConfig = localStorage.getItem("config");
        if (savedConfig) {
            config = JSON.parse(savedConfig);
        }
    });

    let taskResult: TaskResult | undefined = $state(undefined);

    const onEnd = (result: TaskResult) => {
        taskResult = result;
    };

    // config の変更を検知して localStorage に自動保存
    $effect(() => {
        localStorage.setItem("config", JSON.stringify(config));
    });
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
        <ResultDisplay bind:result={taskResult} bind:config />
    {/if}
</main>
