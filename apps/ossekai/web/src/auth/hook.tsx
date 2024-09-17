import { createContext, useContext } from "react";
import type { State, App } from ".";

export type ContextType = [State, App];

export const Context = createContext<ContextType | undefined>(undefined);

export function use(): ContextType {
	const context = useContext(Context);
	if (context === undefined) {
		throw new Error("auth.use must be used within a auth.Provider");
	}
	return context;
}
