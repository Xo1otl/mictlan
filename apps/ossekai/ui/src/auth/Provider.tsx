import { useState, useEffect, type ReactNode } from "react";
import { Outlet } from "@tanstack/react-router";
import { type State, Context, App, AwsCognito, machine } from ".";
import * as libstate from "lib/state";

interface ProviderProps {
	fallback: ReactNode;
}

// Suspendで書きたい
export function Provider({ fallback }: ProviderProps) {
	const [app, setApp] = useState<App | undefined>(undefined);
	const [isLoading, setIsLoading] = useState(true);
	const [state, setState] = useState<State>("unauthenticated");

	console.log("Providerをレンダリングしますよ");

	useEffect(() => {
		console.log("Appを生成しますよ");
		const initializeAuth = async () => {
			// initial stateはsessionをaggregateして導出しよう、applicationの状態を保存すると不整合の元になる
			const iamService = new AwsCognito();

			let initialState: State = "unauthenticated";
			if (await iamService.user()) {
				initialState = "authenticated";
			}
			const stateMachine = new libstate.XState(machine, initialState);
			const newApp = new App(iamService, stateMachine);
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
		<Context.Provider value={[state, app]}>
			<Outlet />
		</Context.Provider>
	);
}
