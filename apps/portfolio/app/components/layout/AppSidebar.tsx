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
	useSidebar,
} from "~/components/ui/sidebar";
import { Link } from "@remix-run/react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useLayoutContext } from "./context";
import type { RouteNode } from "~/hooks/useSiteData";

const LinkTreeItem: React.FC<{ routeNode: RouteNode }> = ({ routeNode }) => {
	const [isOpen, setIsOpen] = useState(false);
	const { toggleSidebar, isMobile } = useSidebar();
	const hasChildren = Boolean(
		routeNode.children && routeNode.children.length > 0,
	);

	return (
		<>
			<SidebarMenuItem>
				<SidebarMenuButton
					onClick={() => setIsOpen(!isOpen)}
					className="flex items-center gap-2"
				>
					{hasChildren &&
						(isOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />)}
					{routeNode.url ? (
						<Link
							to={routeNode.url}
							className="flex-1 hover:underline"
							onClick={() => {
								if (isMobile) {
									toggleSidebar();
								}
							}}
						>
							{routeNode.title}
						</Link>
					) : (
						<span className="flex-1">{routeNode.title}</span>
					)}
				</SidebarMenuButton>
			</SidebarMenuItem>
			{hasChildren && isOpen && (
				<div className="ml-4">
					<SidebarMenu>
						{routeNode.children?.map((child) => (
							<LinkTreeItem key={child.title} routeNode={child} />
						))}
					</SidebarMenu>
				</div>
			)}
		</>
	);
};

export const AppSidebar: React.FC = () => {
	const { routeNode } = useLayoutContext();
	return (
		<Sidebar>
			<SidebarContent>
				<SidebarGroup>
					<SidebarGroupLabel>
						<Link to="/blog">Mictlan's Blog!</Link>
					</SidebarGroupLabel>
					<SidebarGroupContent>
						<SidebarMenu>
							{routeNode.children?.map((routeNode) => (
								<LinkTreeItem key={routeNode.title} routeNode={routeNode} />
							))}
						</SidebarMenu>
					</SidebarGroupContent>
				</SidebarGroup>
			</SidebarContent>
		</Sidebar>
	);
};
