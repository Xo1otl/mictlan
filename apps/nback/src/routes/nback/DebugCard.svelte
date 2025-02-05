<script lang="ts">
    const { value } = $props();
</script>

<main>
    <h3 class="mt-4 text-xl font-bold">デバッグ情報</h3>
    <pre
        class="mt-2 p-2 bg-gray-100 rounded border border-gray-200 shadow">{JSON.stringify(
            value,
            null,
            2,
        )}</pre>
</main>
