import { vitePlugin as remix } from "@remix-run/dev";
import { defineConfig } from "vite";
import tsconfigPaths from "vite-tsconfig-paths";
import mdx from "@mdx-js/rollup";
import rehypeKatex from "rehype-katex";
import { flatRoutes } from "remix-flat-routes";

declare module "@remix-run/node" {
	interface Future {
		v3_singleFetch: true;
	}
}

export default defineConfig({
	plugins: [
		mdx({
			rehypePlugins: [rehypeKatex],
		}),
		remix({
			routes(defineRoutes) {
				const routesData = flatRoutes("routes", defineRoutes, {
					ignoredRouteFiles: ["**/.*"],
				});
				Bun.write(
					"app/assets/routes.json",
					JSON.stringify(routesData, null, 2),
				);
				return routesData;
			},
			future: {
				v3_fetcherPersist: true,
				v3_relativeSplatPath: true,
				v3_throwAbortReason: true,
				v3_singleFetch: true,
				v3_lazyRouteDiscovery: true,
			},
		}),
		tsconfigPaths(),
	],
});
