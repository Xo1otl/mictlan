import type * as statemachine from "..";
import { createActor, type AnyStateMachine } from "xstate";

export class Adapter<State extends statemachine.StateBase, Event extends string>
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

	dispatch<T>(event: Event, effect: () => T): T;
	dispatch<T>(event: Event, effect: () => Promise<T>): Promise<T>;
	dispatch<T>(event: Event, effect: () => T | Promise<T>): T | Promise<T> {
		if (!this.actor.getSnapshot().can({ type: event })) {
			throw new Error(`event '${event}' is not allowed`);
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
