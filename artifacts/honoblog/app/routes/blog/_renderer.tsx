import { jsxRenderer } from "hono/jsx-renderer";
import { DrawerContent } from "../../islands/DrawerContent";
import { DrawerHeadline } from "../../islands/DrawerHeadline";
import { TopAppBar } from "../../islands/TopAppBar";
import { Layout } from "../../islands/Layout";

// interface PageNode {
// 	name: string;
// 	path?: string;
// 	children?: PageNode[];
// }

// const blogDir = "app/routes/blog";
// const tree: PageNode[] = [];
// const glob = new Bun.Glob(`${blogDir}/**/[^_]*`);

// (async () => {
// 	for await (const file of glob.scan(".")) {
// 		const rel = file.slice(blogDir.length + 1).replace(/\.[^/.]+$/, "");
// 		const parts = rel.split("/");
// 		let cur = tree;
// 		for (let i = 0; i < parts.length; i++) {
// 			const p = parts[i];
// 			if (i === parts.length - 1) {
// 				cur.push({ name: p, path: rel });
// 			} else {
// 				let node = cur.find((n) => n.name === p);
// 				if (!node) {
// 					node = { name: p, children: [] };
// 					cur.push(node);
// 				}
// 				// biome-ignore lint/style/noNonNullAssertion: <explanation>
// 				cur = node.children!;
// 			}
// 		}
// 	}
// 	console.log(JSON.stringify(tree, null, 2));
// })();

export default jsxRenderer(({ children, Layout: ParentLayout }) => {
	return (
		<ParentLayout title="Mictlan Blog">
			<Layout
				drawerContent={<DrawerContent pageTree={undefined} />}
				drawerHeadline={<DrawerHeadline />}
				topAppBar={<TopAppBar />}
				appContent={children}
			/>
		</ParentLayout>
	);
});
