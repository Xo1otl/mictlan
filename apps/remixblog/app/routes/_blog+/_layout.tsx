import { type MetaFunction, Outlet } from "@remix-run/react";
import { allPages } from "~/components/layout/pages";
import { SidebarLayout } from "~/components/layout/SidebarLayout";

export const meta: MetaFunction = () => {
	return [
		{ title: "New Remix App" },
		{ name: "description", content: "Welcome to Remix!" },
	];
};

export default function BlogLayout() {
	return (
		<SidebarLayout pages={allPages}>
			<Outlet />
		</SidebarLayout>
	);
}
