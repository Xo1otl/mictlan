import { AppSidebar } from "~/components/layout/AppSidebar";
import { SidebarProvider, SidebarTrigger } from "~/components/ui/sidebar";
import { LayoutProvider } from "./context";
import type { RouteNode } from "~/hooks/useSiteMeta";

type SidebarLayoutProps = {
	children: React.ReactNode;
	routeNode: RouteNode;
};

export const SidebarLayout: React.FC<SidebarLayoutProps> = ({
	children,
	routeNode,
}) => {
	return (
		<LayoutProvider value={{ routeNode }}>
			<SidebarProvider>
				<AppSidebar />
				<main className="prose max-w-none w-full">
					<SidebarTrigger className="fixed" />
					<div className="container mx-auto px-4 py-8">{children}</div>
				</main>
			</SidebarProvider>
		</LayoutProvider>
	);
};
