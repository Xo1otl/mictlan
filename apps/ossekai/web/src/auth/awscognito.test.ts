import { expect, test } from "bun:test";
import { AwsCognito } from "./awscognito";
import * as auth from ".";
import { make } from "lib/utilitytypes";
import testcredentials from "./testcredentials.json";

test("aws cognito", async () => {
	// ログインしてないからユーザーいない
	const iamService = new AwsCognito();
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
