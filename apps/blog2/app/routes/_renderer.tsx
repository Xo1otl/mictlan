import { Style } from "hono/css";
import { jsxRenderer } from "hono/jsx-renderer";
import { Script } from "honox/server";
import { Link } from "honox/server";

export default jsxRenderer(({ children, title }) => {
	return (
		<html lang="en">
			<head>
				<meta charset="utf-8" />
				<meta name="viewport" content="width=device-width, initial-scale=1.0" />
				<Link href="/app/style.css" rel="stylesheet" />
				<title>{title}</title>
				<link rel="icon" href="/favicon.ico" />
				<Script src="/app/client.ts" async />
				<Style />
			</head>
			<body class="prose">{children}</body>
		</html>
	);
});
