import { createFileRoute } from "@tanstack/react-router";
import * as common from "@/src/common";

export const Route = createFileRoute("/_auth/_protected/_layout")({
	component: () => <common.Layout />,
});
