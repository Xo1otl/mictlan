import { machine } from ".";
import type * as auth from "../../auth";
import * as xstatemachine from "pkg/xstatemachine";

export function newStateMachine(initialState: auth.State): auth.StateMachine {
  return new xstatemachine.Adapter(machine, initialState);
}
