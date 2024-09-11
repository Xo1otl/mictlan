import type { Name } from "./entities";

// これはinterface adapter

export interface Library<Notebook> {
  notebook(name: Name): Promise<Notebook>;
}

export interface Presenter<Notebook, RenderResult> {
  render(notebook: Notebook): RenderResult;
}

// TODO: こういうの使えば、進捗表示できそう
export interface Channel<Data> {
  send(data: Data): Promise<void>;
  recieve(): Promise<Data | undefined>;
  close(): void;
}
