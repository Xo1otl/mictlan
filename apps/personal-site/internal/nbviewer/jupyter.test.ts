import { test } from "bun:test";
import { JupyterPresenter, type JupyterNotebook } from "./jupyter";
import { make } from "../../../../packages/lib/ts/pkg/utilitytypes";

test("jupyter render", async () => {
	const presenter = new JupyterPresenter();
	const text = await Bun.file("web/notebook/leanmemo.ipynb").text();
	const notebook = make<JupyterNotebook>();
	const output = await presenter.render(notebook(text));
	console.log("html: ", output);
});
