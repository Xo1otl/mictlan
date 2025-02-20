import honox from "honox/vite";
import tailwindcss from "@tailwindcss/vite";
import mdx from "@mdx-js/rollup";
import remarkFrontmatter from "remark-frontmatter";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { defineConfig, type UserConfig } from "vite";
import ssg from "@hono/vite-ssg";

const entry = "./app/server.ts";

export default defineConfig({
	plugins: [
		honox({
			devServer: { entry },
			client: { input: ["/app/style.css"] },
		}),
		ssg({ entry }),
		tailwindcss() as UserConfig["plugins"],
		mdx({
			jsxImportSource: "hono/jsx",
			rehypePlugins: [rehypeKatex],
			remarkPlugins: [remarkFrontmatter, remarkMdxFrontmatter, remarkMath],
		}),
	],
	server: {
		host: "0.0.0.0",
	},
});
