import type { Name } from "./entities";
import type { Library, Presenter } from "./adapters";

export class App<Notebook, RenderResult> {
	constructor(
		private library: Library<Notebook>,
		private presenter: Presenter<Notebook, RenderResult>,
	) {
		this.library = library;
		this.presenter = presenter;
	}

	// TODO: 進捗表示をしてみたい
	async show(name: Name): Promise<RenderResult> {
		const notebook = await this.library.notebook(name);
		// interfaceとしてChannelを作ってる、これをDIして進捗に応じてChannel.send()とかすればいけそう
		// this.channel.send(progress)
		return this.presenter.render(notebook);
	}
}
