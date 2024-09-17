import { createFileRoute } from "@tanstack/react-router";
import { Outlet, Navigate } from "@tanstack/react-router";
import * as auth from "@/src/auth";

export const Route = createFileRoute("/_auth/_protected")({
	component: () => {
		const [state] = auth.use();
		if (state === "authenticated") {
			return <Outlet />;
		}
		if (state === "unauthenticated") {
			return <Navigate to="/signin" />;
		}
		if (state === "pendingConfirmation") {
			return <Navigate to="/confirm" />;
		}
	},
});
