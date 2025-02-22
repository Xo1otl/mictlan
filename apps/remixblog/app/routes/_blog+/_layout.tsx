import { type MetaFunction, Outlet } from "@remix-run/react";
import { AppSidebar } from "~/components/AppSidebar";
import { SidebarProvider, SidebarTrigger } from "~/components/ui/sidebar";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export default function BlogLayout() {
	return (
		<SidebarProvider>
			<AppSidebar />
			<main className="prose max-w-none w-full">
				<SidebarTrigger />
				<div className="container mx-auto px-4 py-2">
					<Outlet />
				</div>
			</main>
		</SidebarProvider>
	);
}
