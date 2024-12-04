<script lang="ts">
    const { scoreRecord } = $props();
</script>

<div class="scoreboard mt-4 p-4 border rounded-md">
    <h2 class="text-lg font-semibold mb-2">Scoreboard</h2>
    <div class="grid grid-cols-2 gap-2">
        <div>Coordinates:</div>
        <div>
            {scoreRecord.coordinates.matches} / {scoreRecord.coordinates.total}
        </div>
        <div>Character:</div>
        <div>
            {scoreRecord.character.matches} / {scoreRecord.character.total}
        </div>
        <div>Color:</div>
        <div>{scoreRecord.color.matches} / {scoreRecord.color.total}</div>
        <div>Shape:</div>
        <div>{scoreRecord.shape.matches} / {scoreRecord.shape.total}</div>
        <div>Total:</div>
        <div>
            {scoreRecord.total.matches} / {scoreRecord.total.total}
        </div>
    </div>
</div>
