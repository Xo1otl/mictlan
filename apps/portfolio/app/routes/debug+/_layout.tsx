import { type MetaFunction, Outlet } from "@remix-run/react";

export default function BlogLayout() {
	return (
		<div className="prose">
			<Outlet />
		</div>
	);
}
