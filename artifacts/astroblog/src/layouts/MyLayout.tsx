import type { FC, ReactNode } from "react";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar, type Page } from "@/components/AppSidebar";

type MyLayoutProps = {
	children: ReactNode;
	pages: Page[];
};

export const MyLayout: FC<MyLayoutProps> = ({ children, pages }) => {
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
