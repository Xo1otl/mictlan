import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/vendor/shadcn/components/ui/button";
import {
	Sheet,
	SheetContent,
	SheetTrigger,
} from "@/vendor/shadcn/components/ui/sheet";
import { Menu, Home, FileText, MessageCircle } from "lucide-react";
import { UserMenu } from "../../../components/UserMenu";
import { Link, Outlet } from "@tanstack/react-router";

export const Layout = () => {
	const NavLinks = () => (
		<>
			<Button asChild variant="ghost" className="justify-start">
				<Link to="/home">
					<Home className="mr-2 h-4 w-4" />
					Home
				</Link>
			</Button>
			<Button asChild variant="ghost" className="justify-start">
				<Link to="/debug">
					<FileText className="mr-2 h-4 w-4" />
					Debug
				</Link>
			</Button>
			<Button asChild variant="ghost" className="justify-start">
				<Link to="/demo">
					<MessageCircle className="mr-2 h-4 w-4" />
					Demo
				</Link>
			</Button>
		</>
	);

	return (
		<div className="flex h-screen">
			{/* Sidebar for larger screens */}
			<aside className="hidden lg:flex w-64 flex-col p-4">
				<nav className="flex flex-col space-y-2">
					<NavLinks />
				</nav>
			</aside>

			{/* Main content area */}
			<div className="flex flex-col flex-1">
				{/* Navbar */}
				<header>
					<div className="flex items-center justify-between p-4">
						<div className="flex items-center">
							<Sheet>
								<SheetTrigger asChild>
									<Button
										variant="outline"
										size="icon"
										className="lg:hidden mr-4"
									>
										<Menu />
									</Button>
								</SheetTrigger>
								<SheetContent side="left" className="w-64 p-0">
									<nav className="flex flex-col space-y-2 p-4">
										<NavLinks />
									</nav>
								</SheetContent>
							</Sheet>
							<h1 className="text-xl font-semibold">Page Title</h1>
						</div>
						<div className="flex items-center space-x-4">
							<Button>Ask Question</Button>
							<UserMenu />
						</div>
					</div>
				</header>

				{/* Page content */}
				<main className="flex-1 overflow-auto p-4">
					<Outlet />
				</main>
			</div>
		</div>
	);
};

export const Route = createFileRoute("/_auth/_protected/_layout")({
	component: () => <Layout />,
});
