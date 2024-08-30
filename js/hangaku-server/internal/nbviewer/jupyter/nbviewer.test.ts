import { test } from "bun:test";
import { Presenter, type Notebook } from "./nbviewer";
import { make } from "pkg/utilitytypes";

test("jupyter render", async () => {
  const presenter = new Presenter();
  const text = await Bun.file("web/notebook/leanmemo.ipynb").text();
  const notebook = make<Notebook>();
  const output = await presenter.render(notebook(text));
  console.log("html: ", output);
});
