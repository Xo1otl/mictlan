import { createRoute } from "honox/factory";

export default createRoute((c) => {
	return c.render(
		<div>
			<h1>Hello, World!</h1>
		</div>,
	);
});
