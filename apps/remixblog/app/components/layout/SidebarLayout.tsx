import { AppSidebar } from "~/components/layout/AppSidebar";
import { SidebarProvider, SidebarTrigger } from "~/components/ui/sidebar";
import type { Page } from "./pages";

export type SidebarLayoutProps = {
	children: React.ReactNode;
	pages: Page[];
};

export const SidebarLayout: React.FC<SidebarLayoutProps> = ({
	children,
	pages,
}) => {
	return (
		<SidebarProvider>
			<AppSidebar pages={pages} />
			<main className="prose max-w-none w-full">
				<SidebarTrigger />
				<div className="container mx-auto px-4 py-2">{children}</div>
			</main>
		</SidebarProvider>
	);
};
