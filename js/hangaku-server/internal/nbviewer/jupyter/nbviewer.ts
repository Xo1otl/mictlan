import type * as nbviewer from "..";
import { type Brand, make } from "pkg/utilitytypes";
import { join } from "node:path";

// NotebookがviewModel
export type Notebook = Brand<string, "Notebook">;
export type HTML = Brand<string, "HTML">;

export class Library implements nbviewer.Library<Notebook> {
	async notebook(name: nbviewer.Name): Promise<Notebook> {
		if (!/^[a-zA-Z0-9_-]+$/.test(name)) {
			throw Error("invalid notebook name");
		}
		const fullpath = join("web/notebook/", `${name}.ipynb`);
		const file = Bun.file(fullpath);
		if (await file.exists()) {
			const notebook = make<Notebook>();
			const noteText = await file.text();
			return notebook(noteText);
		}
		throw Error("notebook not found");
	}
}

export class Presenter implements nbviewer.Presenter<Notebook, Promise<HTML>> {
	async render(notebook: Notebook): Promise<HTML> {
		try {
			const respose = new Response(notebook);
			const output =
				await Bun.$`jupyter nbconvert --to html --execute --stdin --stdout < ${respose}`.text();
			const html = make<HTML>();
			return html(output);
		} catch (error) {
			console.error("Failed to convert notebook:", error);
			throw error;
		}
	}
}
