import { createFileRoute } from "@tanstack/react-router";
import { Outlet, Navigate } from "@tanstack/react-router";
import { useAuth } from "../../hooks/useAuth";

export const Route = createFileRoute("/_auth/_protected")({
	component: () => {
		const [state] = useAuth();
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
