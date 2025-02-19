import { type FC, useState, type Child } from "hono/jsx";

type SidebarLayoutProps = {
	sidebarContent: Child;
	mainContent: Child;
};

export const SidebarLayout: FC<SidebarLayoutProps> = ({
	sidebarContent,
	mainContent,
}) => {
	const [sidebarOpen, setSidebarOpen] = useState(true);
	const [drawerOpen, setDrawerOpen] = useState(false);

	return (
		<div class="relative min-h-screen flex">
			{/* デスクトップサイドバー */}
			<div
				class={`hidden md:flex flex-col bg-gray-200 transition-all duration-300 ${
					sidebarOpen ? "w-64" : "w-0 overflow-hidden"
				}`}
			>
				{/* 高さ固定のヘッダー領域 */}
				<div class="h-12 flex items-center p-2">
					<button
						type="button"
						class="p-2 hover:bg-gray-300 rounded"
						onClick={() => setSidebarOpen(false)}
						aria-label="Collapse sidebar"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="h-5 w-5"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<title>Collapse sidebar</title>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M15 19l-7-7 7-7"
							/>
						</svg>
					</button>
				</div>
				<div class="flex-1 overflow-y-auto p-2">{sidebarContent}</div>
			</div>

			{/* メインコンテンツ */}
			<div class="flex-1">
				{/* ヘッダー領域（全ケース共通） */}
				<div class="h-12 flex items-center p-2">
					{/* デスクトップ用開閉ボタン */}
					{!sidebarOpen && (
						<button
							type="button"
							class="hidden md:inline-flex p-2 hover:bg-gray-300 rounded"
							onClick={() => setSidebarOpen(true)}
							aria-label="Expand sidebar"
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								class="h-5 w-5"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<title>Expand sidebar</title>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="2"
									d="M9 5l7 7-7 7"
								/>
							</svg>
						</button>
					)}

					{/* モバイル用ハンバーガーボタン */}
					<button
						type="button"
						class="md:hidden p-2 hover:bg-gray-300 rounded"
						onClick={() => setDrawerOpen(true)}
						aria-label="Open sidebar"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="h-6 w-6"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<title>Open sidebar</title>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M4 6h16M4 12h16M4 18h16"
							/>
						</svg>
					</button>
				</div>

				{/* メインコンテンツ本体 */}
				<div class="p-2">{mainContent}</div>
			</div>

			{/* モバイルドロワー */}
			<div
				class={`fixed inset-0 z-40 bg-black/50 transition-opacity duration-300 md:hidden ${
					drawerOpen ? "opacity-100" : "opacity-0 pointer-events-none"
				}`}
				onClick={() => setDrawerOpen(false)}
				onKeyUp={(e) => {
					if (e.key === "Enter" || e.key === " ") {
						setDrawerOpen(false);
					}
				}}
			/>

			<div
				class={`fixed inset-y-0 left-0 z-50 w-64 bg-gray-200 transition-transform duration-300 md:hidden ${
					drawerOpen ? "translate-x-0" : "-translate-x-full"
				}`}
			>
				{/* ドロワーヘッダー */}
				<div class="h-12 flex items-center justify-between p-2">
					<button
						type="button"
						class="p-2 hover:bg-gray-300 rounded"
						onClick={() => setDrawerOpen(false)}
						aria-label="Close sidebar"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="h-5 w-5"
							fill="none"
							viewBox="0 0 24 24"
							stroke="currentColor"
						>
							<title>Close sidebar</title>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M6 18L18 6M6 6l12 12"
							/>
						</svg>
					</button>
				</div>
				<div class="flex-1 overflow-y-auto p-2">{sidebarContent}</div>
			</div>
		</div>
	);
};
