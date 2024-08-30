import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/_auth/_protected/_layout/home")({
	component: () => <div>Hello /_auth/_protected/_layout/home!</div>,
});
