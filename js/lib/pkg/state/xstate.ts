/**
 * @file
 * @description
 * xstateによるmachineの実装
 * clean architectureではinterface adapter(プログラミング言語のinterfaceではない)の役割
 */
import type { State, Machine, Event } from "./machine";
import { createActor, type AnyStateMachine } from "xstate";

export class XState<T extends State, U extends Event> implements Machine<T, U> {
	private actor;

	constructor(machine: AnyStateMachine, initialState: T) {
		this.actor = createActor(machine, {
			state: machine.resolveState({
				value: initialState,
				context: undefined,
			}),
		});
		this.actor.start();
	}

	subscribe(listener: (state: T) => void) {
		return this.actor.subscribe((state) => listener(state.value));
	}

	dispatch<T>(event: U, effect?: () => T): T;
	dispatch<T>(event: U, effect?: () => Promise<T>): Promise<T>;
	dispatch<T>(
		event: U,
		effect?: () => T | Promise<T>,
	): T | Promise<T> | undefined {
		if (!this.actor.getSnapshot().can({ type: event })) {
			throw new Error(`event '${event}' is not allowed`);
		}

		if (!effect) {
			this.actor.send({ type: event });
			return;
		}

		const result = effect();
		if (result instanceof Promise) {
			return result.then((value) => {
				this.actor.send({ type: event });
				return value;
			});
		}
		this.actor.send({ type: event });
		return result;
	}
}
