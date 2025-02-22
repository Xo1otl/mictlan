import type React from "react";
import type { Page } from "./AppSidebar";

type NavigationProps = {
	pages: Page[];
};

export const Navigation: React.FC<NavigationProps> = ({ pages }) => {
	console.log(pages);

	return <>{pages}</>;
};
