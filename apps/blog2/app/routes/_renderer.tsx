import { Style } from "hono/css";
import { jsxRenderer } from "hono/jsx-renderer";
import { Script } from "honox/server";
import { Link } from "honox/server";
import { SidebarLayout } from "../islands/SidebarLayout";

export default jsxRenderer(({ children, title }) => {
	return (
		<html lang="en">
			<head>
				<meta charSet="utf-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1.0" />
				<Link href="/app/style.css" rel="stylesheet" />
				<title>{title}</title>
				<link rel="icon" href="/favicon.ico" />
				<Script src="/app/client.ts" async />
				<Style />
			</head>
			{/* proseがmax-widthを謎に指定するのでそれを無効化しないとやたら狭い領域にしかmdxが表示されない */}
			<body class="prose" style="max-width: none !important;">
				<SidebarLayout sidebarContent={<h1>AAA</h1>} mainContent={children} />
			</body>
		</html>
	);
});
