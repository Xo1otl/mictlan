import { createContext, useContext } from "react";

type Theme = {
	drawerBgClass: string;
	appBgClass: string;
};

export const themes = {
	default: {
		drawerBgClass: "bg-gray-200",
		appBgClass: "bg-white",
	},
};

export const ThemeContext = createContext<Theme | undefined>(undefined);

export const useTheme = () => {
	const context = useContext(ThemeContext);
	if (!context) {
		throw new Error("useTheme must be used within a ThemeProvider");
	}
	return context;
};
