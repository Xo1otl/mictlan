import { type MetaFunction, Outlet } from "@remix-run/react";
import { SidebarLayout } from "~/components/layout/SidebarLayout";
import { useSiteMeta } from "~/hooks/useSiteMeta";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export default function BlogLayout() {
	const siteMeta = useSiteMeta();
	const blogNode = siteMeta.blogRoute();
	return (
		<SidebarLayout routeNode={blogNode}>
			<Outlet />
		</SidebarLayout>
	);
}
