import build from "@hono/vite-build/cloudflare-pages";
import adapter from "@hono/vite-dev-server/cloudflare";
import honox from "honox/vite";
import tailwindcss from "@tailwindcss/vite";
import mdx from "@mdx-js/rollup";
import remarkFrontmatter from "remark-frontmatter";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { defineConfig, type UserConfig } from "vite";

export default defineConfig({
	plugins: [
		honox({ devServer: { adapter }, client: { input: ["/app/style.css"] } }),
		build(),
		tailwindcss() as UserConfig["plugins"],
		mdx({
			jsxImportSource: "hono/jsx",
			rehypePlugins: [rehypeKatex],
			remarkPlugins: [remarkFrontmatter, remarkMdxFrontmatter, remarkMath],
		}),
	],
});
