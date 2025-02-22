// @ts-check
import { defineConfig } from "astro/config";

import react from "@astrojs/react";

import mdx from "@astrojs/mdx";
import rehypeKatex from "rehype-katex";

import tailwindcss from "@tailwindcss/vite";

import node from "@astrojs/node";

// https://astro.build/config
export default defineConfig({
	integrations: [react(), mdx()],

	markdown: {
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
