import { createContext, useContext } from "react";
import type { RouteNode } from "~/hooks/useSiteMeta";

export type LayoutContext = {
	routeNode: RouteNode;
};

const LayoutContext = createContext<LayoutContext | undefined>(undefined);

export const LayoutProvider = LayoutContext.Provider;

export const useLayoutContext = () => {
	const context = useContext(LayoutContext);
	if (!context) {
		throw new Error("useLayoutContext must be used within a LayoutProvider");
	}
	return context;
};
