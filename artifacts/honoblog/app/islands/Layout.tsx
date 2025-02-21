import { useState, type Child, type FC } from "hono/jsx";
import { ThemeContext, themes, useTheme } from "../hooks/theme";

type LayoutProps = {
	drawerHeadline: Child;
	drawerContent: Child;
	topAppBar: Child;
	appContent: Child;
};

export const Layout: FC<LayoutProps> = ({
	drawerHeadline,
	drawerContent,
	topAppBar,
	appContent,
}) => {
	const [drawerOpen, setDrawerOpen] = useState(true);
	const [drawerModalOpen, setDrawerModalOpen] = useState(false);
	return (
		<div class="prose lg:prose-xl max-w-none relative min-h-screen flex">
			<ThemeContext.Provider value={themes.default}>
				{/* md以上の場合、drawerOpenがtrueのときだけ Drawer をレンダリング */}
				{drawerOpen && (
					<Drawer
						onClose={() => setDrawerOpen(false)}
						headline={drawerHeadline}
						content={drawerContent}
					/>
				)}

				{/* small screens用: drawerModalOpenがtrueのときだけ ModalDrawer をレンダリング */}
				{drawerModalOpen && (
					<ModalDrawer
						onClose={() => setDrawerModalOpen(false)}
						headline={drawerHeadline}
						content={drawerContent}
					/>
				)}

				{/* メインの App エリア */}
				<App
					topBar={topAppBar}
					content={appContent}
					drawerOpen={drawerOpen}
					onDrawerOpen={() => setDrawerOpen(true)}
					onModalDrawerOpen={() => setDrawerModalOpen(true)}
				/>
			</ThemeContext.Provider>
		</div>
	);
};

type DrawerProps = {
	onClose: () => void;
	headline: Child;
	content: Child;
};

const Drawer: FC<DrawerProps> = ({ onClose, headline, content }) => {
	const theme = useTheme();
	return (
		<div class={`hidden md:flex flex-col ${theme.drawerBgClass} w-64`}>
			<div class="h-12 flex items-center p-2">
				<HambergurButton onClick={onClose} />
				<div class="ml-2 flex items-center h-full">{headline}</div>
			</div>
			<div class="flex-1 overflow-y-auto p-2">{content}</div>
		</div>
	);
};

type ModalDrawerProps = {
	onClose: () => void;
	headline: Child;
	content: Child;
};

const ModalDrawer: FC<ModalDrawerProps> = ({ onClose, headline, content }) => {
	const theme = useTheme();
	return (
		<>
			<div
				class={`fixed inset-y-0 left-0 z-50 w-64 ${theme.drawerBgClass} md:hidden`}
			>
				<div class="h-12 flex items-center p-2">
					<HambergurButton onClick={onClose} />
					<div class="ml-2 flex items-center h-full">{headline}</div>
				</div>
				<div class="flex-1 overflow-y-auto p-2">{content}</div>
			</div>
			<div
				class="fixed inset-0 z-40 bg-black/50 md:hidden"
				onClick={onClose}
				onKeyUp={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						onClose();
					}
				}}
			/>
		</>
	);
};

type AppProps = {
	topBar: Child;
	content: Child;
	drawerOpen: boolean;
	onDrawerOpen: () => void;
	onModalDrawerOpen: () => void;
};

const App: FC<AppProps> = ({
	topBar,
	content,
	drawerOpen,
	onDrawerOpen,
	onModalDrawerOpen,
}) => {
	const theme = useTheme();
	return (
		<div class={`flex-1 ${theme.appBgClass}`}>
			<div class="h-12 flex items-center p-2">
				{/* md以上: Drawerが閉じている場合のみ表示 */}
				{!drawerOpen && (
					<div class="hidden md:inline-flex">
						<HambergurButton onClick={onDrawerOpen} />
					</div>
				)}
				{/* small screens: ModalDrawer を開くためのボタン */}
				<div class="md:hidden">
					<HambergurButton onClick={onModalDrawerOpen} />
				</div>
				<div class="flex-1 flex items-center h-full ml-2">{topBar}</div>
			</div>
			<div class="p-2">{content}</div>
		</div>
	);
};

type HambergurButtonProps = {
	onClick: () => void;
};

const HambergurButton: FC<HambergurButtonProps> = ({ onClick }) => (
	<button
		type="button"
		class="p-2 hover:bg-gray-300 rounded"
		onClick={onClick}
		aria-label="Toggle navigation drawer"
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			class="h-6 w-6"
			fill="none"
			viewBox="0 0 24 24"
			stroke="currentColor"
		>
			<title>Toggle navigation drawer</title>
			<path
				strokeLinecap="round"
				strokeLinejoin="round"
				strokeWidth="2"
				d="M4 6h16M4 12h16M4 18h16"
			/>
		</svg>
	</button>
);
