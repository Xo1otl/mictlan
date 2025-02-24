import { type MetaFunction, Outlet } from "@remix-run/react";
import { SidebarLayout } from "~/components/layout/SidebarLayout";
import { useSiteData } from "~/hooks/useSiteData";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export default function BlogLayout() {
	const siteData = useSiteData();
	const blogNode = siteData.blogNode();
	return (
		<SidebarLayout routeNode={blogNode}>
			<Outlet />
		</SidebarLayout>
	);
}
