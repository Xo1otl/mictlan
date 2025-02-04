import { expect, test, describe } from "bun:test";
import {
	newTaskEngine,
	newTrialFactory,
	type MatchResult,
	type Trial,
	type TrialResult,
} from "./task";

describe("TaskEngine Tests", () => {
	const trialFactory = newTrialFactory({
		stimulusTypes: ["color", "shape", "position"],
		colors: ["red"],
		gridSize: [5, 5],
	});

	// Dummy readTrialInput that always returns a fixed input
	const readTrialInput = (): MatchResult[] => {
		return [
			{ stimulusType: "color", match: true },
			{ stimulusType: "shape", match: false },
			{ stimulusType: "position", match: false },
		];
	};

	// Test 1: Verify that the engine generates the correct number of trials
	test("generates expected number of trials", async () => {
		const n = 2;
		const problemCount = 3; // Total iterations should be n + problemCount = 5
		const interval = 10; // in milliseconds

		// Array to capture generated trials via onGenerateTrial callback
		const generatedTrials: Trial[] = [];

		// onGenerateTrial callback: push the generated trial into generatedTrials
		const onUpdate = (newTrial: Trial): void => {
			generatedTrials.push(newTrial);
		};

		// Create the engine instance
		const engine = newTaskEngine(n, problemCount, interval, trialFactory);

		// Start the engine
		engine.start(readTrialInput, onUpdate);

		// Wait enough time for all trials to be generated.
		// Total iterations = n + problemCount = 5, so wait a bit longer than 5*interval.
		await new Promise((resolve) => setTimeout(resolve, interval * 6));

		// The engine should have generated exactly 5 trials.
		expect(generatedTrials.length).toBe(n + problemCount);
	});

	// Test 2: Verify that calling reset stops further trial generation.
	test("reset stops the engine", async () => {
		const n = 2;
		const problemCount = 10; // Use a higher count to observe the reset effect
		const interval = 10;

		let callCount = 0;
		const generatedTrials: Trial[] = [];

		// onGenerateTrial callback increases the call count and stores the trial
		const onUpdate = (newTrial: Trial, prevTrialResult?: TrialResult): void => {
			callCount++;
			generatedTrials.push(newTrial);
			console.debug("prevTrialResult", prevTrialResult);
		};

		const engine = newTaskEngine(n, problemCount, interval, trialFactory);

		const reset = engine.start(readTrialInput, onUpdate);

		// Wait for a short duration (e.g. 5 intervals)
		await new Promise((resolve) => setTimeout(resolve, interval * 5));

		// Call reset to stop the engine
		reset();

		// Store the call count at reset time
		const countAtReset = callCount;

		// Wait longer to ensure no additional trials are generated after reset
		await new Promise((resolve) => setTimeout(resolve, interval * 5));

		// The call count should remain the same after reset
		expect(callCount).toBe(countAtReset);
	});
});
