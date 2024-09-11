import { createContext, useContext } from "react";
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
