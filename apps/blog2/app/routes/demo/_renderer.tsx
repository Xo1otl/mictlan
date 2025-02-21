import { jsxRenderer } from "hono/jsx-renderer";
import { DrawerContent } from "../../islands/DrawerContent";
import { DrawerHeadline } from "../../islands/DrawerHeadline";
import { TopAppBar } from "../../islands/TopAppBar";
import { Layout } from "../../islands/Layout";
import { Demo } from "../../islands/Demo";

export default jsxRenderer(({ children, Layout: ParentLayout }) => {
	return (
		<ParentLayout title="Mictlan Blog">
			{/* <Layout
				drawerContent={<DrawerContent pageTree={undefined} />}
				drawerHeadline={<DrawerHeadline />}
				topAppBar={<TopAppBar />}
				appContent={children}
			/> */}
			<Demo>{children}</Demo>
		</ParentLayout>
	);
});
