import type * as statemachine from "pkg/statemachine";
import type {
  Event,
  Password,
  Code,
  State,
  User,
  Username,
  Token,
} from "./entities";

export interface IAMService {
  user(): Promise<User | undefined>;
  token(): Promise<Token | undefined>;
  signIn(username: Username, password: Password): Promise<void>;
  signOut(): Promise<void>;
  signUp(username: Username, password: Password): Promise<void>;
  confirm(code: Code): Promise<void>;
}

export type StateMachine = statemachine.InterfaceAdapter<State, Event>;
