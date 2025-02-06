import { elasticOut } from "svelte/easing";

export const spin = (node: Element, { duration }: { duration: number }) => {
	return {
		duration,
		css: (t: number, u: number) => {
			const eased = elasticOut(t);
			return `transform: scale(${eased}) rotate(${eased * 360}deg);`;
		},
	};
};
