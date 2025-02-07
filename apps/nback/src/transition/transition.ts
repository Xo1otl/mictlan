import { elasticOut } from "svelte/easing";

export const spin = (node: Element, { duration }: { duration: number }) => {
	return {
		duration,
		css: (t: number, u: number) => {
			const minScale = 0.1;
			const eased = elasticOut(t);
			const scaleValue = minScale + (1 - minScale) * eased;
			return `transform: scale(${scaleValue}) rotate(${eased * 360}deg);`;
		},
	};
};
