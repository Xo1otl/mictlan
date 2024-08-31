import { User, LogOut, Settings, HelpCircle } from "lucide-react";
import { useAuth } from "../hooks/auth";
import {
	DropdownMenu,
	DropdownMenuContent,
	DropdownMenuItem,
	DropdownMenuTrigger,
} from "@/vendor/shadcn/components/ui/dropdown-menu";
import { Button } from "@/vendor/shadcn/components/ui/button";

export const UserMenu = () => {
	const [, app] = useAuth();

	const handleLogout = (): void => {
		app.signOut();
	};

	const handleSettings = (): void => {
		console.log("Opening settings...");
		// ここに設定画面を開く処理を実装
	};

	const handleHelp = (): void => {
		console.log("Opening help...");
		// ここにヘルプ画面を開く処理を実装
	};

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button variant="ghost" size="icon" className="rounded-full">
					<User className="h-5 w-5" />
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="end">
				<DropdownMenuItem onClick={handleSettings}>
					<Settings className="mr-2 h-4 w-4" />
					<span>Settings</span>
				</DropdownMenuItem>
				<DropdownMenuItem onClick={handleHelp}>
					<HelpCircle className="mr-2 h-4 w-4" />
					<span>Help</span>
				</DropdownMenuItem>
				<DropdownMenuItem onClick={handleLogout}>
					<LogOut className="mr-2 h-4 w-4" />
					<span>Log out</span>
				</DropdownMenuItem>
			</DropdownMenuContent>
		</DropdownMenu>
	);
};
