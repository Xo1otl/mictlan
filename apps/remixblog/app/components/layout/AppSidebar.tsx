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
import { Link } from "@remix-run/react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { Page } from "./pages";

const PageTreeItem: React.FC<{ page: Page }> = ({ page }) => {
	const [isOpen, setIsOpen] = useState(false);
	const hasChildren = Boolean(page.children && page.children.length > 0);

	return (
		<>
			<SidebarMenuItem>
				<SidebarMenuButton
					onClick={() => setIsOpen(!isOpen)}
					className="flex items-center gap-2"
				>
					{hasChildren &&
						(isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />)}
					{page.url ? (
						<Link to={page.url} className="flex-1">
							{page.title}
						</Link>
					) : (
						<span className="flex-1">{page.title}</span>
					)}
				</SidebarMenuButton>
			</SidebarMenuItem>
			{hasChildren && isOpen && (
				<div className="ml-4">
					<SidebarMenu>
						{page.children?.map((child) => (
							<PageTreeItem key={child.title} page={child} />
						))}
					</SidebarMenu>
				</div>
			)}
		</>
	);
};

type AppSidebarProps = {
	pages: Page[];
};

export const AppSidebar: React.FC<AppSidebarProps> = ({ pages }) => {
	return (
		<Sidebar>
			<SidebarContent>
				<SidebarGroup>
					<SidebarGroupLabel>Mictlan's Blog!</SidebarGroupLabel>
					<SidebarGroupContent>
						<SidebarMenu>
							{pages.map((page) => (
								<PageTreeItem key={page.title} page={page} />
							))}
						</SidebarMenu>
					</SidebarGroupContent>
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
};
