import type { IAMService } from "./adapters";
import type { StateMachine } from "./adapters";
import type { Password, Code, State, Username } from "./entities";

export class App {
	constructor(
		private iamService: IAMService,
		private stateMachine: StateMachine,
	) {}

	async user() {
		const user = await this.iamService.user();
		if (user) {
			return user;
		}
		await this.signOut();
	}

	async token() {
		const token = await this.iamService.token();
		if (token) {
			return token;
		}
		await this.signOut();
	}

	async signIn(username: Username, password: Password) {
		await this.stateMachine.dispatch("signIn", async () => {
			await this.iamService.signIn(username, password);
		});
	}

	async signOut() {
		await this.stateMachine.dispatch("signOut", async () => {
			await this.iamService.signOut();
		});
	}

	async signUp(username: Username, password: Password) {
		await this.stateMachine.dispatch("signUp", async () => {
			await this.iamService.signUp(username, password);
		});
	}

	async confirm(code: Code) {
		await this.stateMachine.dispatch("confirm", async () => {
			await this.iamService.confirm(code);
		});
	}

	subscribe(listener: (state: State) => void): () => void {
		return this.stateMachine.subscribe(listener);
	}
}
