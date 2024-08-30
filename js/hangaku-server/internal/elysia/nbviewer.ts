import Elysia from "elysia";
import * as nbviewer from "../nbviewer";
import * as jupyter from "../nbviewer/jupyter";
import { make } from "pkg/utilitytypes";

export function launchNbviewer(port: number) {
	const app = new nbviewer.App(new jupyter.Library(), new jupyter.Presenter());
	new Elysia()
		.get(
			"nbviewer/:name",
			({ params: { name } }) => {
				const nbName = make<nbviewer.Name>();
				return app.show(nbName(name));
			},
			{
				afterHandle({ response, set }) {
					set.headers["content-type"] = "text/html;charset=utf8";
				},
			},
		)
		.get("*", () => {
			return "404 not found";
		})
		.listen(port);
}
