import { AppSidebar } from "~/components/layout/AppSidebar";
import { SidebarProvider, SidebarTrigger } from "~/components/ui/sidebar";
import { LayoutProvider } from "./context";
import type * as sitedata from "~/shared";

type SidebarLayoutProps = {
	children: React.ReactNode;
	routeNode: sitedata.RouteNode;
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
					<SidebarTrigger />
					<div className="container mx-auto px-4 py-2">{children}</div>
				</main>
			</SidebarProvider>
		</LayoutProvider>
	);
};
