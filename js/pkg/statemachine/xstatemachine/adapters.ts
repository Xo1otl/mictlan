import type * as statemachine from "..";
import { createActor, type AnyStateMachine } from "xstate";

export class Adapter<State extends string, Event extends string>
	implements statemachine.InterfaceAdapter<State, Event>
{
	private actor;

	constructor(machine: AnyStateMachine, initialState: State) {
		this.actor = createActor(machine, {
			state: machine.resolveState({
				value: initialState,
				context: undefined,
			}),
		});
		this.actor.start();
	}

	subscribe(listener: (state: State) => void) {
		const subscription = this.actor.subscribe((state) => listener(state.value));
		return subscription.unsubscribe;
	}

	transition<T>(event: Event, action: () => T): T;
	transition<T>(event: Event, action: () => Promise<T>): Promise<T>;
	transition<T>(event: Event, action: () => T | Promise<T>): T | Promise<T> {
		if (!this.actor.getSnapshot().can({ type: event })) {
			throw new Error(`event '${event}' is not allowed`);
		}

		const result = action();
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
