import { type MetaFunction, Outlet } from "@remix-run/react";
import { SidebarLayout } from "~/components/layout/SidebarLayout";
import * as shared from "~/shared";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export default function BlogLayout() {
	const siteContext = shared.getSiteContext();
	return (
		<SidebarLayout routeNode={siteContext.routeNode()}>
			<Outlet />
		</SidebarLayout>
	);
}
