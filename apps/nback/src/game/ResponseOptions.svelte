<script lang="ts">
	import type { MatchResult } from "./game";
	const { manager } = $props();
	let buttonStates: MatchResult = $state({
		color: false,
		shape: false,
		coordinates: false,
		character: false,
	});
	manager.subscribe((event: string) => {
		if (event === "update") {
			buttonStates = {
				color: false,
				shape: false,
				coordinates: false,
				character: false,
			};
			manager.input(buttonStates);
		}
		console.log("buttonStates", buttonStates.shape);
	});

	let responseOptions = [
		{
			label: "色",
			value: "color",
			bgColor: "bg-red-500",
			textColor: "text-white",
			onBgColor: "bg-red-700",
			offBgColor: "bg-red-300",
		},
		{
			label: "形",
			value: "shape",
			bgColor: "bg-blue-500",
			textColor: "text-white",
			onBgColor: "bg-blue-700",
			offBgColor: "bg-blue-300",
		},
		{
			label: "文字",
			value: "character",
			bgColor: "bg-green-500",
			textColor: "text-white",
			onBgColor: "bg-green-700",
			offBgColor: "bg-green-300",
		},
		{
			label: "場所",
			value: "coordinates",
			bgColor: "bg-indigo-600",
			textColor: "text-white",
			onBgColor: "bg-indigo-800",
			offBgColor: "bg-indigo-300",
		},
	];

	function handleClick(
		value: "color" | "shape" | "coordinates" | "character",
	) {
		buttonStates[value] = !buttonStates[value];
		manager.input(buttonStates);
	}
</script>

<div class="flex justify-center space-x-2">
	{#each responseOptions as option (option.value)}
		<button
			class="px-4 py-2 rounded-md {buttonStates[
				option.value as 'color' | 'shape' | 'coordinates' | 'character'
			]
				? option.onBgColor
				: option.offBgColor} {option.textColor}"
			onclick={() =>
				handleClick(
					option.value as
						| "color"
						| "shape"
						| "coordinates"
						| "character",
				)}
		>
			{option.label}
		</button>
	{/each}
</div>
