import React from "react";
import routes from "~/assets/routes.json";

export const useSiteMeta = (): SiteContext => {
	const routes = React.useMemo(buildRoutes, []);
	const blogRoutes = routes.children?.find((child) => {
		return child.title === "blog";
	});
	if (!blogRoutes) {
		throw new Error("blog route not found");
	}
	return {
		blogRoute() {
			return blogRoutes;
		},
	};
};

export type RouteNode = {
	title: string;
	url?: string;
	children?: RouteNode[];
};

export interface SiteContext {
	blogRoute(): RouteNode;
}

const buildRoutes = (): RouteNode => {
	const flatNodes = Object.values(routes);

	// 1. マップとルートノードの初期化
	const nodeMap = new Map<string, RouteNode>();
	const root: RouteNode = { title: "root", children: [] };
	nodeMap.set("/", root);

	// 2. ツリー構築処理
	for (const node of flatNodes) {
		const segments = node.id.split("/").slice(1); // `routes`の部分を除去

		let currentPath = "";
		let parent = root;

		// パスをセグメントごとに処理
		for (let i = 0; i < segments.length; i++) {
			let segment = segments[i];

			// _indexは親のURLを設定する
			if (segment.startsWith("_index")) {
				parent.url = currentPath;
				continue;
			}

			//　無視するセグメント
			if (segment.startsWith("_")) {
				continue;
			}

			const isLeaf = i === segments.length - 1;

			// 末尾が+のセグメント
			if (segment.endsWith("+")) {
				segment = segment.slice(0, -1);
			}

			currentPath += `/${segment}`;

			// 既存ノードを探す
			let route = nodeMap.get(currentPath);

			if (!route) {
				// 新規ノード作成
				route = {
					title: segment,
					children: [],
				};

				// マップに登録
				nodeMap.set(currentPath, route);

				// 親ノードに追加
				if (parent.children) {
					parent.children.push(route);
				} else {
					parent.children = [route];
				}
			}

			// 最後のセグメントの場合のみ元データをマージ
			if (isLeaf) {
				Object.assign(route, {
					url: currentPath,
					title: segment,
				});
			}

			// 次の反復用に親を更新
			parent = route;
		}
	}

	return root;
};
