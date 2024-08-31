import { createContext, useCallback, useContext } from "react";
import type * as auth from "../../auth";

export type AuthContextType = [auth.State, auth.App];

export const AuthContext = createContext<AuthContextType | undefined>(
	undefined,
);

export function useAuth(): AuthContextType {
	const context = useContext(AuthContext);
	if (context === undefined) {
		throw new Error("useAuth must be used within a AuthProvider");
	}
	return context;
}

// TODO: ここにapi等を登録して、それをurl直書きではなくapiの名前で呼び出せるようにする
export function useAuthenticatedFetch(): typeof fetch {
	const [, app] = useAuth();

	return useCallback(
		async (input, init) => {
			const token = await app.token();

			return fetch(input, {
				...init,
				headers: {
					...init?.headers,
					Authorization: `Bearer ${token}`,
				},
			});
		},
		[app],
	);
}
