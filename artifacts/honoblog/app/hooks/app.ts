import { createContext, useContext } from "hono/jsx";


type AppContextType = {
	currentPage: string;
};

export const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
	const context = useContext(AppContext);
	if (!context) {
		throw new Error("useApp must be used within a AppProvider");
	}
	return context;
};
