import { jsxRenderer } from "hono/jsx-renderer";
import { Demo } from "../../islands/Demo";

export default jsxRenderer(({ children, Layout: ParentLayout }) => {
	return (
		<ParentLayout title="Mictlan Blog">
			<Demo>{children}</Demo>
		</ParentLayout>
	);
});
