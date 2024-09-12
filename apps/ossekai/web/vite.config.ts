import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "node:path";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [
		react(),
		TanStackRouterVite({
			routesDirectory: "./internal/ui/routes",
			generatedRouteTree: "./internal/ui/routeTree.gen.ts",
		}),
	],
	resolve: {
		alias: {
			"@": path.resolve(__dirname, "."),
		},
	},
});