import routes from "~/assets/routes.json";

export type Page = {
	title: string;
	url?: string;
	children?: Page[];
};

// 必要なフィールドのみを定義した RoutesEntry 型
export type RoutesEntry = {
	id: string;
	parentId: string;
	path?: string;
	index?: boolean;
};

export const allPages: Page[] = (() => {
	// 中間構造として利用する Node 型
	type Node = {
		title: string;
		url?: string;
		children: Map<string, Node>;
	};

	// ルートレベルのノードを保持する Map
	const rootMap = new Map<string, Node>();

	// ヘルパー関数：パスの各セグメントを tree に挿入
	const insertPath = (segments: string[], url: string) => {
		let currentMap = rootMap;
		for (let i = 0; i < segments.length; i++) {
			const seg = segments[i];
			if (!currentMap.has(seg)) {
				currentMap.set(seg, { title: seg, children: new Map<string, Node>() });
			}
			const node = currentMap.get(seg);
			if (!node) {
				throw new Error(`Node not found for segment: ${seg}`);
			}
			// 最後のセグメントの場合、url を設定
			if (i === segments.length - 1) {
				node.url = url;
			}
			currentMap = node.children;
		}
	};

	// routes を走査
	for (const key in routes) {
		const route = routes[key as keyof typeof routes] as RoutesEntry;
		if (route.path) {
			// path を '/' で分割し、先頭が '_' で始まるセグメントは除外
			const segments = route.path
				.split("/")
				.filter((seg) => !seg.startsWith("_"));
			if (segments.length > 0) {
				insertPath(segments, route.path);
			}
		} else {
			// path がない場合
			if (route.index === true) {
				// ルートの index ページは home として扱い、url は '/'
				if (!rootMap.has("home")) {
					rootMap.set("home", {
						title: "home",
						url: "/",
						children: new Map<string, Node>(),
					});
				}
			}
			// それ以外は無視
		}
	}

	// Map を Page 配列に再帰変換する関数
	const convertMapToPages = (map: Map<string, Node>): Page[] => {
		const pages: Page[] = [];
		for (const node of map.values()) {
			const page: Page = { title: node.title };
			if (node.url) {
				page.url = node.url;
			}
			const childrenPages = convertMapToPages(node.children);
			if (childrenPages.length > 0) {
				page.children = childrenPages;
			}
			pages.push(page);
		}
		return pages;
	};

	return convertMapToPages(rootMap);
})();
