import { createFileRoute } from "@tanstack/react-router";
import * as auth from "../auth";

export const Route = createFileRoute("/_auth")({
	component: () => <auth.Provider fallback={<p>Loading...</p>} />,
});
