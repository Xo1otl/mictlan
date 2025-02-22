import type React from "react";
import { useState } from "react";
import {
	Sidebar,
	SidebarContent,
	SidebarGroup,
	SidebarGroupContent,
	SidebarGroupLabel,
	SidebarMenu,
	SidebarMenuButton,
	SidebarMenuItem,
} from "~/components/ui/sidebar";
import routes from "~/assets/routes.json";

// 再帰的なページ情報の型定義
export type Page = {
	title: string;
	// urlが指定されている場合はリンクとして扱う
	url?: string;
	// 子要素がある場合は、再帰的にPageの配列を持つ
	children?: Page[];
};

type AppSidebarProps = {
	// 再帰的なpages構造
	pages: Page[];
};

// 単一のナビゲーション項目を再帰的にレンダリングするコンポーネント
const SidebarItem: React.FC<{ page: Page }> = ({ page }) => {
	// 子要素を持つ場合、展開状態を管理するためのstateを定義
	const [isOpen, setIsOpen] = useState(false);
	const hasChildren = Boolean(page.children && page.children.length > 0);

	if (hasChildren) {
		return (
			<div>
				{/* 子要素がある場合は、クリックで展開／折りたたみできるボタンを表示 */}
				<SidebarMenuItem>
					<SidebarMenuButton onClick={() => setIsOpen(!isOpen)}>
						{/* 子要素展開用のインジケーターとして「▶」「▼」を利用 */}
						<span>{page.title}</span>
						<span style={{ marginLeft: "auto" }}>{isOpen ? "▼" : "▶"}</span>
					</SidebarMenuButton>
				</SidebarMenuItem>
				{isOpen && (
					// 子要素はインデントして表示するため、余白用のラッパーを使用
					<div className="ml-4">
						<SidebarMenu>
							{page.children?.map((child) => (
								<SidebarItem key={child.title} page={child} />
							))}
						</SidebarMenu>
					</div>
				)}
			</div>
		);
	}
	return (
		<SidebarMenuItem>
			<SidebarMenuButton asChild>
				<a href={page.url}>
					<span>{page.title}</span>
				</a>
			</SidebarMenuButton>
		</SidebarMenuItem>
	);
};

// AppSidebarコンポーネントは、pages配列を受け取りSidebar内に再帰的にレンダリングする
export const AppSidebar: React.FC = () => {
	console.log(routes);
	return (
		<Sidebar>
			<SidebarContent>
				<SidebarGroup>
					<SidebarGroupLabel>Application</SidebarGroupLabel>
					<SidebarGroupContent>
						<SidebarMenu>
							{/* {routes.map((page) => (
								<SidebarItem key={page.title} page={page} />
							))} */}
						</SidebarMenu>
					</SidebarGroupContent>
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
};
