import Elysia from "elysia";
import * as nbviewer from "../nbviewer";
import * as utilitytypes from "../../../../packages/lib/ts/pkg/utilitytypes";

// コマンドライン引数からポート番号を取得する関数
function getPortFromArgs(): number {
	const args = process.argv.slice(2);
	let port = 4002; // デフォルト値

	for (let i = 0; i < args.length; i++) {
		if (args[i] === "--port" && i + 1 < args.length) {
			const portValue = Number.parseInt(args[i + 1], 10);
			if (!Number.isNaN(portValue)) {
				port = portValue;
				break;
			}
		}
	}

	return port;
}

export function launchElysia() {
	// ポート番号を取得
	const port = getPortFromArgs();

	const app = new nbviewer.App(
		new nbviewer.JupyterLibrary(),
		new nbviewer.JupyterPresenter(),
	);

	new Elysia()
		.get(
			"nbviewer/:name",
			({ params: { name } }) => {
				const nbName = utilitytypes.make<nbviewer.Name>();
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

	console.log(`Listening on http://localhost:${port} ...`);
}
