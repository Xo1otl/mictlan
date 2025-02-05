<script lang="ts">
    // Svelte 5 using runes: use $props() instead of `export let`
    import type { TrialResult } from "../../nback";
    let { results } = $props<{ results: TrialResult[] }>();
</script>

<section class="result-list">
    <h2>結果</h2>
    {#if results.length === 0}
        <p>まだ結果はありません。</p>
    {:else}
        <ul>
            {#each results as result}
                <li>
                    <strong>トライアル {result.trial_idx}:</strong>
                    <ul>
                        {#each result.matchResults as matchResult}
                            <li>
                                {matchResult.stimulusType}: {matchResult.match
                                    ? "○"
                                    : "×"}
                            </li>
                        {/each}
                    </ul>
                </li>
            {/each}
        </ul>
    {/if}
</section>

<style>
    .result-list {
        width: 100%;
        max-width: 600px;
        margin: 1rem auto;
    }
    .result-list ul {
        list-style: none;
        padding: 0;
    }
    .result-list li {
        margin: 0.5rem 0;
        padding: 0.25rem;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
</style>
