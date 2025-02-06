<script lang="ts">
    import * as nback from "../../nback/index";

    const { trial }: { trial: nback.Trial } = $props();
    const stimuli = trial.stimuli();

    const color = stimuli.color ? stimuli.color : nback.Color.Black;
    const shape = stimuli.shape ? stimuli.shape : nback.Shape.Square;
    const character = stimuli.character;
</script>

<!-- グロー効果用のフィルター定義 -->
<svg width="0" height="0">
    <defs>
        <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
            </feMerge>
        </filter>
    </defs>
</svg>

<div class="flex flex-col items-center justify-center w-full h-full">
    {#if shape === nback.Shape.Circle}
        <svg class="w-full h-full" viewBox="0 0 100 100">
            <!-- 円形 -->
            <circle
                cx="50"
                cy="50"
                r="35"
                fill="none"
                stroke={color}
                stroke-width="4"
                filter="url(#glow)"
            />
            <text
                x="50"
                y="52"
                text-anchor="middle"
                dominant-baseline="middle"
                fill={color}
                font-size="24"
                font-weight="bold"
            >
                {character}
            </text>
        </svg>
    {:else if shape === nback.Shape.Square}
        <svg class="w-full h-full" viewBox="0 0 100 100">
            <!-- 正方形 -->
            <rect
                x="15"
                y="15"
                width="70"
                height="70"
                fill="none"
                stroke={color}
                stroke-width="4"
                filter="url(#glow)"
            />
            <text
                x="50"
                y="52"
                text-anchor="middle"
                dominant-baseline="middle"
                fill={color}
                font-size="24"
                font-weight="bold"
            >
                {character}
            </text>
        </svg>
    {:else if shape === nback.Shape.Triangle}
        <svg class="w-full h-full" viewBox="0 0 100 100">
            <!-- 三角形 -->
            <polygon
                points="50,20 85,80 15,80"
                fill="none"
                stroke={color}
                stroke-width="4"
                filter="url(#glow)"
            />
            <text
                x="50"
                y="62"
                text-anchor="middle"
                dominant-baseline="middle"
                fill={color}
                font-size="24"
                font-weight="bold"
            >
                {character}
            </text>
        </svg>
    {:else if shape === nback.Shape.Pentagon}
        <svg class="w-full h-full" viewBox="0 0 100 100">
            <!-- 正五角形 -->
            <polygon
                points="50,15 83.29,39.18 70.57,78.32 29.43,78.32 16.71,39.18"
                fill="none"
                stroke={color}
                stroke-width="4"
                filter="url(#glow)"
            />
            <text
                x="50"
                y="52"
                text-anchor="middle"
                dominant-baseline="middle"
                fill={color}
                font-size="24"
                font-weight="bold"
            >
                {character}
            </text>
        </svg>
    {/if}
</div>
