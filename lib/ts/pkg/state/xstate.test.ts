import { test } from "bun:test";
import { setup } from "xstate";
import { XState } from "./xstate";

const testMachine = setup({
	types: {
		context: {} as object,
		events: {} as
			| { type: "EnterFirstName" }
			| { type: "EnterLastName" }
			| { type: "EnterEmail" }
			| { type: "EnterPassword" },
	},
}).createMachine({
	context: {},
	id: "SignUpForm",
	type: "parallel",
	states: {
		Email: {
			initial: "Invalid",
			states: {
				Invalid: {
					on: {
						EnterEmail: {
							target: "Valid",
						},
					},
				},
				Valid: {},
			},
		},
		Password: {
			initial: "Invalid",
			states: {
				Invalid: {
					on: {
						EnterPassword: {
							target: "Valid",
						},
					},
				},
				Valid: {},
			},
		},
		Name: {
			type: "parallel",
			states: {
				FirstName: {
					initial: "Invalid",
					states: {
						Invalid: {
							on: {
								EnterFirstName: {
									target: "Valid",
								},
							},
						},
						Valid: {},
					},
				},
				LastName: {
					initial: "Invalid",
					states: {
						Invalid: {
							on: {
								EnterLastName: {
									target: "Valid",
								},
							},
						},
						Valid: {},
					},
				},
			},
		},
	},
});

type TestState = {
	Name: {
		FirstName: "Invalid" | "Valid";
		LastName: "Invalid" | "Valid";
	};
	Email: "Invalid" | "Valid";
	Password: "Invalid" | "Valid";
};

test("xstate", () => {
	const initialState: TestState = {
		Name: {
			FirstName: "Invalid",
			LastName: "Invalid",
		},
		Email: "Invalid",
		Password: "Invalid",
	};
	const machine = new XState(testMachine, initialState);
	machine.subscribe((state) => {
		console.log(`subscribed ${JSON.stringify(state)}`);
	});
	machine.dispatch("EnterEmail");
	const output = machine.dispatch("EnterPassword", () => {
		console.log("this is effect");
		return "effect output";
	});
	console.log(output);
});
