import { createRoute } from "honox/factory";

export default createRoute((c) => {
	const name = c.req.query("name") ?? "Hono";
	return c.render(
		<div>
			<h1>Hello, {name}!</h1>
		</div>,
		{ title: `Hello, ${name}!` },
	);
});
