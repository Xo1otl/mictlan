import { jsxRenderer } from "hono/jsx-renderer";
import { DrawerContent } from "../../islands/DrawerContent";
import { DrawerHeadline } from "../../islands/DrawerHeadline";
import { TopAppBar } from "../../islands/TopAppBar";
import { Layout } from "../../islands/Layout";

export default jsxRenderer(({ children, Layout: ParentLayout }) => {
	return (
		<ParentLayout title="Mictlan Blog">
			<Layout
				drawerContent={<DrawerContent />}
				drawerHeadline={<DrawerHeadline />}
				topAppBar={<TopAppBar />}
				appContent={children}
			/>
		</ParentLayout>
	);
});
