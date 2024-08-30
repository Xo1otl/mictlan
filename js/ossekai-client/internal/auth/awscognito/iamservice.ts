import * as auth from "..";
import { Amplify } from "aws-amplify";
import {
  confirmSignUp,
  getCurrentUser,
  fetchAuthSession,
  signIn,
  signOut,
  signUp,
} from "aws-amplify/auth";
import awsconfig from "./awsconfig.json";
import { make } from "pkg/utilitytypes";

export class IAMService implements auth.IAMService {
  private pendingConfirmationUsername: auth.Username | undefined;

  constructor() {
    Amplify.configure(awsconfig);
  }

  async user() {
    const session = await fetchAuthSession();
    if (!session.tokens) return;
    const username = (await getCurrentUser())?.signInDetails?.loginId;
    if (!username) return;

    return { username: new auth.Username(username) };
  }

  async token() {
    const session = await fetchAuthSession();
    if (!session.tokens) return;
    const token = make<auth.Token>();
    return token(session.tokens.accessToken.toString());
  }

  async signIn(username: auth.Username, password: auth.Password) {
    const user = await signIn({ username: username.toString(), password });
    if (user.isSignedIn) {
      console.log("サインイン成功:", user);
    }
  }

  async signOut() {
    await signOut();
    console.log("サインアウト成功");
  }

  async signUp(username: auth.Username, password: auth.Password) {
    // TODO: フォーム作って入力受け取るようにする
    await signUp({
      username: username.toString(),
      password,
      options: {
        userAttributes: {
          family_name: "spicy",
          given_name: "island",
        },
      },
    });
    this.pendingConfirmationUsername = username;
  }

  async confirm(code: auth.Code) {
    if (!this.pendingConfirmationUsername) {
      throw new Error("username not set");
    }
    const { isSignUpComplete } = await confirmSignUp({
      username: this.pendingConfirmationUsername.toString(),
      confirmationCode: code,
    });
    console.log(isSignUpComplete);
  }
}
