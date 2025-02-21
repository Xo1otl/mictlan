// @ts-check
import { defineConfig } from "astro/config";

import react from "@astrojs/react";

import mdx from "@astrojs/mdx";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import tailwindcss from "@tailwindcss/vite";

import node from "@astrojs/node";

// https://astro.build/config
export default defineConfig({
	integrations: [react(), mdx()],

	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},

	vite: {
		plugins: [tailwindcss()],
	},

	adapter: node({
		mode: "standalone",
	}),

	server: {
		host: "0.0.0.0",
	},
});
