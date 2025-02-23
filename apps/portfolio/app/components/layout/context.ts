import { createContext, useContext } from "react";
import type * as sitedata from "~/shared";

export type LayoutContext = {
	routeNode: sitedata.RouteNode;
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
