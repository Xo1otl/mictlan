import { useState, useEffect, type ReactNode } from "react";
import { createFileRoute, Outlet } from "@tanstack/react-router";
import { AuthContext } from "../hooks/auth";
import * as auth from "../../auth";
import * as libstate from "lib/state";

interface AuthProviderProps {
	fallback: ReactNode;
}

// Suspendで書きたい
export function AuthProvider({ fallback }: AuthProviderProps) {
	const [app, setApp] = useState<auth.App | undefined>(undefined);
	const [isLoading, setIsLoading] = useState(true);
	const [state, setState] = useState<auth.State>("unauthenticated");

	console.log("Providerをレンダリングしますよ");

	useEffect(() => {
		console.log("Appを生成しますよ");
		const initializeAuth = async () => {
			// initial stateはsessionをaggregateして導出しよう、applicationの状態を保存すると不整合の元になる
			const iamService = new auth.AwsCognito();

			let initialState: auth.State = "unauthenticated";
			if (await iamService.user()) {
				initialState = "authenticated";
			}
			const stateMachine = new libstate.XState(auth.machine, initialState);
			const newApp = new auth.App(iamService, stateMachine);
			setState(initialState);
			newApp.subscribe((state) => setState(state));
			setApp(newApp);
			setIsLoading(false);
		};
		initializeAuth();
	}, []);

	if (isLoading || !app) {
		return fallback;
	}

	return (
		<AuthContext.Provider value={[state, app]}>
			<Outlet />
		</AuthContext.Provider>
	);
}

export const Route = createFileRoute("/_auth")({
	component: () => <AuthProvider fallback={<p>Loading...</p>} />,
});
