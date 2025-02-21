import type { FC, ReactNode } from "react";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";

export const MyLayout: FC<{ children: ReactNode }> = ({
	children,
}: { children: ReactNode }) => {
	return (
		<SidebarProvider>
			<AppSidebar />
			<main className="prose max-w-none w-full">
				<SidebarTrigger />
				<div className="container mx-auto px-4 py-2">{children}</div>
			</main>
		</SidebarProvider>
	);
};
