import { expect, test } from "bun:test";
import { IAMService } from "./iamservice";
import * as auth from "..";
import { make } from "pkg/utilitytypes";
import testcredentials from "./testcredentials.json";

test("iam service", async () => {
	// ログインしてないからユーザーいない
	const iamService = new IAMService();
	let user = await iamService.user();
	expect(user).toBeUndefined();

	// ログインしたからユーザーがいる
	const username = new auth.Username(testcredentials.username);
	const password = make<auth.Password>();
	await iamService.signIn(username, password(testcredentials.password));
	user = await iamService.user();
	expect(user).not.toBeUndefined();

	// サインアウトしたからユーザーいない
	await iamService.signOut();
	user = await iamService.user();
	expect(user).toBeUndefined();
});
