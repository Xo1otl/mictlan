import * as email from "lib/email";
import type { Brand } from "lib/utilitytypes";

// emailであることを表すバリデーション等
export class Username extends email.ValueObject {}
export type Token = Brand<string, "Token">;
export type Password = Brand<string, "Password">;
export type Code = Brand<string, "Code">;

export type User = {
	username: Username;
};

export type State = "unauthenticated" | "authenticated" | "pendingConfirmation";
export type Event = "signIn" | "signUp" | "confirm" | "signOut";
