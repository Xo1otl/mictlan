import honox from "honox/vite";
import tailwindcss from "@tailwindcss/vite";
import { compile, type CompileOptions } from "@mdx-js/mdx";
import type { Plugin } from "vite";
import remarkFrontmatter from "remark-frontmatter";
import remarkMdxFrontmatter from "remark-mdx-frontmatter";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { defineConfig } from "vite";
import ssg from "@hono/vite-ssg";
import client from "honox/vite/client";

const entry = "./app/server.ts";

const mdx = (opts: Readonly<CompileOptions>): Plugin => {
	return {
		name: "mdx-island",
		async transform(source, id) {
			if (id.endsWith(".mdx")) {
				console.log(opts);
				const code = await compile(source, opts);
				if (id.includes("demo3")) {
					console.log("===== DEMO3 =====");
					console.log(code.value);
					code.value = `\
					import {jsx as _jsx, jsxs as _jsxs} from "hono/jsx/jsx-runtime";
					export const frontmatter = undefined;
					function _createMdxContent(props) {
					  const _components = {
					    annotation: "annotation",
					    math: "math",
					    mi: "mi",
					    mrow: "mrow",
					    semantics: "semantics",
					    span: "span",
					    ...props.components
					  };
					  return _jsx(_components.math, {
					    className: "katex-display",
					    children: _jsxs(_components.h1, {children: "A"})
					  });
					}
					export default function MDXContent(props = {}) {
					  const {wrapper: MDXLayout} = props.components || ({});
					  return MDXLayout ? _jsx(MDXLayout, {
					    ...props,
					    children: _jsx(_createMdxContent, {
					      ...props
					    })
					  }) : _createMdxContent(props);
					}
										`;
				}
				if (id.includes("demo2")) {
					console.log("===== DEMO2 =====");
					console.log(code.value);
				}

				return { code: code.value.toString() };
			}
		},
	};
};

export default defineConfig(({ mode }) => {
	if (mode === "client") {
		return {
			plugins: [client({ input: ["/app/style.css"] }), tailwindcss()],
		};
	}
	return {
		build: {
			emptyOutDir: false,
		},
		plugins: [
			honox({
				devServer: { entry },
				client: { input: ["/app/style.css"] },
			}),
			tailwindcss(),
			ssg({ entry }),
			mdx({
				jsxImportSource: "hono/jsx",
				rehypePlugins: [rehypeKatex],
				remarkPlugins: [remarkFrontmatter, remarkMdxFrontmatter, remarkMath],
			}),
		],
		server: {
			host: "0.0.0.0",
		},
	};
});
