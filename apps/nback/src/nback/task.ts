export interface TaskEngine {
	start(
		readTrialInput: () => MatchResult[],
		onUpdate?: (newTrial: Trial, prevTrialResult?: TrialResult) => void,
	): () => void;
}

export interface TrialFactory {
	random(): Trial;
}

export interface Trial {
	compare(other: Trial): MatchResult[];
	stimuli(): TrialStimuli;
}

export type TrialResult = {
	trial_idx: number;
	matchResults: MatchResult[];
};

export type MatchResult = {
	stimulusType: keyof TrialStimuli;
	match: boolean;
};

export type TrialStimuli = {
	position: [number, number];
	color?: string;
	character?: string;
	shape?: string;
	sound?: string;
	animation?: string;
};

type TaskState = {
	current_trial_idx: number;
	queue: Trial[];
};

export const newTaskEngine = (
	n: number,
	problemCount: number,
	interval: number,
	trialFactory: TrialFactory,
): TaskEngine => {
	return {
		start(
			readTrialInput: () => MatchResult[],
			onUpdate?: (newTrial: Trial, prevTrialResult?: TrialResult) => void,
		): () => void {
			let state: TaskState = {
				current_trial_idx: 0,
				queue: [],
			};

			let timer: ReturnType<typeof setInterval> | undefined = undefined;

			const reset = () => {
				if (timer !== undefined) {
					clearInterval(timer);
					timer = undefined;
				}
				state = {
					current_trial_idx: 0,
					queue: [],
				};
			};

			const loopTrial = () => {
				let trialResult: TrialResult | undefined = undefined;

				if (state.queue.length === n + 1) {
					const previousTrial = state.queue.shift();
					if (previousTrial === undefined) {
						reset();
						throw new Error("previousTrial is undefined");
					}
					const latestTrial = state.queue[state.queue.length - 1];
					const systemResult = previousTrial.compare(latestTrial);

					const inputResult = readTrialInput();

					const matchResults = systemResult.map((sys) => {
						const input = inputResult.find(
							(inp) => inp.stimulusType === sys.stimulusType,
						);
						if (!input) {
							reset();
							throw new Error(`Input result not found for ${sys.stimulusType}`);
						}
						return {
							stimulusType: sys.stimulusType,
							match: input.match === sys.match,
						};
					});

					trialResult = {
						trial_idx: state.current_trial_idx,
						matchResults: matchResults,
					};

					if (state.current_trial_idx >= problemCount + n) {
						reset();
						return;
					}
				}

				const trial = trialFactory.random();
				state.queue.push(trial);
				state.current_trial_idx++;

				if (onUpdate) {
					onUpdate(trial, trialResult);
				}
			};

			loopTrial();
			timer = setInterval(loopTrial, interval);
			return reset;
		},
	};
};

export type TrialFactoryOptions = {
	stimulusTypes?: (keyof TrialStimuli)[];
	colors?: string[];
	shapes?: string[];
	characters?: string[];
	sounds?: string[];
	animations?: string[];
	gridSize?: [number, number];
};

export const newTrialFactory = ({
	stimulusTypes = ["color", "character", "position"],
	colors = ["red", "green", "blue", "black"],
	shapes = ["circle", "square", "triangle", "star"],
	characters = "ABCDEHKLMO".split(""),
	sounds = ["beep", "boop", "ring"],
	animations = ["spin", "fade", "bounce"],
	gridSize = [3, 3],
}: TrialFactoryOptions): TrialFactory => {
	const random = (): Trial => {
		const stimuli: TrialStimuli = {
			position: [
				Math.floor(Math.random() * gridSize[0]),
				Math.floor(Math.random() * gridSize[1]),
			],
		};
		if (stimulusTypes.includes("color")) {
			stimuli.color = colors[Math.floor(Math.random() * colors.length)];
		}
		if (stimulusTypes.includes("character")) {
			stimuli.character =
				characters[Math.floor(Math.random() * characters.length)];
		}
		if (stimulusTypes.includes("shape")) {
			stimuli.shape = shapes[Math.floor(Math.random() * shapes.length)];
		}
		if (stimulusTypes.includes("sound")) {
			stimuli.sound = sounds[Math.floor(Math.random() * sounds.length)];
		}
		if (stimulusTypes.includes("animation")) {
			stimuli.animation =
				animations[Math.floor(Math.random() * animations.length)];
		}

		return {
			compare(other: Trial): MatchResult[] {
				const results: MatchResult[] = [];
				for (const type of stimulusTypes) {
					const value = stimuli[type];
					const otherValue = other.stimuli()[type];
					results.push({
						stimulusType: type,
						match: value === otherValue,
					});
				}
				return results;
			},
			stimuli(): TrialStimuli {
				return stimuli;
			},
		};
	};

	return {
		random,
	};
};
