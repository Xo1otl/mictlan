<script lang="ts">
    // Svelte 5 using runes: use $props() instead of `export let`
    import type { TrialStimuli } from "../../nback";
    let { trials } = $props<{ trials: TrialStimuli[] }>();
</script>

<section class="trial-list">
    <h2>生成されたトライアル</h2>
    {#if trials.length === 0}
        <p>まだトライアルは生成されていません。</p>
    {:else}
        <ul>
            {#each trials as trial, index}
                <li>
                    <strong>トライアル {index + 1}:</strong>
                    {#if trial.character}
                        <span> キャラクター: {trial.character}</span>
                    {/if}
                    {#if trial.position}
                        <span>
                            - 位置: ({trial.position[0]}, {trial
                                .position[1]})</span
                        >
                    {/if}
                </li>
            {/each}
        </ul>
    {/if}
</section>

<style>
    .trial-list {
        width: 100%;
        max-width: 600px;
        margin: 1rem auto;
    }
    .trial-list ul {
        list-style: none;
        padding: 0;
    }
    .trial-list li {
        margin: 0.5rem 0;
        padding: 0.25rem;
        /* 後にテーマ変更できるよう、境界線や丸みも CSS 変数などで管理可能 */
        border: 1px solid #e0e0e0;
        border-radius: 4px;
    }
</style>
