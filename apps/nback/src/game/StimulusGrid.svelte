<script lang="ts">
  import type { Stimulus } from "./game"; // Stimulus型をインポート

  let { stimulus } = $props();

  const gridSize = 4;

  const x = $derived(stimulus?.coordinates?.x ?? -1);
  const y = $derived(stimulus?.coordinates?.y ?? -1);

  function getCellContent(row: number, col: number): string | undefined {
    if (row === y && col === x) {
      return stimulus?.character.value || ""; // charactorを使う
    }
    return undefined;
  }

  function getCellClass(row: number, col: number): string {
    let classes =
      "border-2 rounded-md p-5 text-2xl text-center flex justify-center items-center min-h-[50px] aspect-square ";

    if (row === y && col === x) {
      classes += "relative "; // shapeを追加するためのrelativeを追加
    }

    return classes;
  }

  const colorMap: {
    [key: string]: { base: string; light: string; medium: string };
  } = {
    red: { base: "#ff0000", light: "#ffe0e0", medium: "#ff8080" },
    green: { base: "#00ff00", light: "#e0ffe0", medium: "#80ff80" },
    blue: { base: "#0000ff", light: "#e0e0ff", medium: "#8080ff" },
    yellow: { base: "#ffff00", light: "#ffffd0", medium: "#ffff80" },
  };

  function getCellStyles(row: number, col: number): string {
    if (row === y && col === x) {
      const color = stimulus?.color?.value;
      if (color && colorMap[color]) {
        const mediumColor = colorMap[color].medium;
        return `background-color: ${mediumColor};`;
      }
    }
    return "";
  }

  function getShape(): string | undefined {
    switch (stimulus.shape.value) {
      case "circle":
        return '<div class="absolute w-16 h-16 rounded-full bg-opacity-50 border-2 border-black inset-1/4"></div>';
      case "square":
        return '<div class="absolute w-16 h-16 bg-opacity-50 border-2 border-black inset-1/4"></div>';
      case "triangle":
        return '<div class="absolute w-16 h-16 transform rotate-45 bg-opacity-50 border-2 border-black inset-1/4 skew-x-12 skew-y-12"></div>'; // 三角形の表現を改善 (回転と歪み)
    }
    return undefined;
  }
</script>

<div class="grid grid-cols-4 gap-2 w-4/5 max-w-md mx-auto mb-5">
  {#each Array(gridSize) as _, row}
    {#each Array(gridSize) as _, col}
      <div class={getCellClass(row, col)} style={getCellStyles(row, col)}>
        {getCellContent(row, col)}
        {#if x === col && y === row}
          {@html getShape()}
        {/if}
      </div>
    {/each}
  {/each}
</div>
