import type * as nbviewer from ".";
import { type Brand, make } from "../../../../packages/lib/ts/pkg/utilitytypes";
import { join } from "node:path";

// NotebookがviewModel
export type JupyterNotebook = Brand<string, "Notebook">;
export type HTML = Brand<string, "HTML">;

export class JupyterLibrary implements nbviewer.Library<JupyterNotebook> {
	async notebook(name: nbviewer.Name): Promise<JupyterNotebook> {
		if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
			throw Error("invalid notebook name");
		}
		const fullpath = join("web/notebook/", `${name}.ipynb`);
		const file = Bun.file(fullpath);
		if (await file.exists()) {
			const notebook = make<JupyterNotebook>();
			const noteText = await file.text();
			return notebook(noteText);
		}
		throw Error("notebook not found");
	}
}

export class JupyterPresenter
	implements nbviewer.Presenter<JupyterNotebook, Promise<HTML>>
{
	async render(notebook: JupyterNotebook): Promise<HTML> {
		try {
			const respose = new Response(notebook);
			// notebookが自分のパスを基準にして外部ファイルを参照することがあるためnotebookが置かれる場所に移動して実行する
			const output =
				await Bun.$`cd web/notebook && jupyter nbconvert --to html --execute --stdin --stdout < ${respose}`.text();
			const html = make<HTML>();
			return html(output);
		} catch (error) {
			console.error("Failed to convert notebook:", error);
			throw error;
		}
	}
}
