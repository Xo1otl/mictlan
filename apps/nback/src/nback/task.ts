export interface TaskEngine {
	start(
		readTrialInput: () => MatchResult[],
		onUpdate?: (newTrial?: Trial, prevTrialResult?: TrialResult) => void,
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
	trialIdx: number;
	cohensKappa: Record<StimulusType, number>;
	matchResults: MatchResult[];
};

export type MatchResult = {
	stimulusType: StimulusType;
	match: boolean;
};

export type TrialStimuli = {
	[key in StimulusType]?: key extends StimulusType.Position
		? [number, number]
		: key extends StimulusType.Color
			? Color
			: key extends StimulusType.Character
				? Character
				: key extends StimulusType.Shape
					? Shape
					: key extends StimulusType.Sound
						? Sound
						: key extends StimulusType.Animation
							? Animation
							: never;
};

export enum StimulusType {
	Position = "position",
	Color = "color",
	Character = "character",
	Shape = "shape",
	Sound = "sound",
	Animation = "animation",
}

export enum Color {
	Red = "red",
	Green = "green",
	Purple = "purple",
	Black = "black",
}

export enum Shape {
	Triangle = "triangle",
	Square = "square",
	Pentagon = "pentagon",
	Circle = "circle",
}

export enum Character {
	ZERO = "0",
	ONE = "1",
	TWO = "2",
	THREE = "3",
	FOUR = "4",
	FIVE = "5",
	SIX = "6",
	SEVEN = "7",
	EIHGT = "8",
	NINE = "9",
	A = "A",
	B = "B",
	C = "C",
	D = "D",
	E = "E",
	H = "H",
	K = "K",
	L = "L",
	M = "M",
	O = "O",
}

export enum Sound {
	A = "A",
	B = "B",
	C = "C",
	H = "H",
	K = "K",
	L = "L",
	M = "M",
	O = "O",
}

export enum Animation {
	Blur = "blur",
	Fly = "fly",
	Scale = "scale",
	Spin = "spin",
	None = "none",
}

export type TrialFactoryOptions = {
	stimulusTypes?: StimulusType[];
	colors?: Color[];
	shapes?: Shape[];
	characters?: Character[];
	sounds?: Sound[];
	animations?: Animation[];
	gridSize?: [number, number];
};

export const DefaultTrialFactoryOptions: TrialFactoryOptions & {
	gridSize: [number, number];
} = {
	stimulusTypes: [
		StimulusType.Color,
		StimulusType.Shape,
		StimulusType.Sound,
		StimulusType.Position,
	],
	colors: Object.values(Color),
	shapes: Object.values(Shape),
	sounds: Object.values(Sound),
	gridSize: [3, 3],
};

type TaskState = {
	currentTrialIdx: number;
	queue: Trial[];
	hit: Record<StimulusType, number>;
	correctRejection: Record<StimulusType, number>;
	falseAlarm: Record<StimulusType, number>;
	miss: Record<StimulusType, number>;
};

const newTaskState = (): TaskState => {
	const defaultValues = {
		[StimulusType.Position]: 0,
		[StimulusType.Color]: 0,
		[StimulusType.Character]: 0,
		[StimulusType.Shape]: 0,
		[StimulusType.Sound]: 0,
		[StimulusType.Animation]: 0,
	};
	return {
		currentTrialIdx: 0,
		queue: [],
		hit: { ...defaultValues },
		correctRejection: { ...defaultValues },
		falseAlarm: { ...defaultValues },
		miss: { ...defaultValues },
	};
};

export type TaskEngineOptions = {
	n: number;
	problemCount: number;
	interval: number;
	trialFactory: TrialFactory;
};

export const newTaskEngine = ({
	n,
	problemCount,
	interval,
	trialFactory,
}: TaskEngineOptions): TaskEngine => {
	// startはステートレスなので何回呼び出しても問題ない
	return {
		start(
			readTrialInput: () => MatchResult[],
			onUpdate?: (newTrial?: Trial, prevTrialResult?: TrialResult) => void,
		): () => void {
			let state = newTaskState();
			let timer: ReturnType<typeof setInterval> | undefined = undefined;

			const reset = () => {
				if (timer !== undefined) {
					clearInterval(timer);
					timer = undefined;
				}
				state = newTaskState();
			};

			const loopTrial = () => {
				let trialResult: TrialResult | undefined = undefined;

				if (state.queue.length === n + 1) {
					const previousTrial = state.queue.shift();
					if (previousTrial === undefined) {
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
							throw new Error(`Input result not found for ${sys.stimulusType}`);
						}
						if (input.match && sys.match) {
							state.hit[input.stimulusType]++;
						}
						if (!sys.match && !input.match) {
							state.correctRejection[sys.stimulusType]++;
						}
						if (!sys.match && input.match) {
							state.falseAlarm[sys.stimulusType]++;
						}
						if (sys.match && !input.match) {
							state.miss[sys.stimulusType]++;
						}
						return {
							stimulusType: sys.stimulusType,
							match: input.match === sys.match,
						};
					});

					let cohensKappa: Record<StimulusType, number> = {
						[StimulusType.Position]: Number.NaN,
						[StimulusType.Color]: Number.NaN,
						[StimulusType.Character]: Number.NaN,
						[StimulusType.Shape]: Number.NaN,
						[StimulusType.Sound]: Number.NaN,
						[StimulusType.Animation]: Number.NaN,
					};

					for (const stimulusType of Object.values(StimulusType)) {
						const hit = state.hit[stimulusType];
						const falseAlarm = state.falseAlarm[stimulusType];
						const miss = state.miss[stimulusType];
						const correctRejection = state.correctRejection[stimulusType];
						cohensKappa[stimulusType] = calculateCohensKappa(
							hit,
							falseAlarm,
							miss,
							correctRejection,
						);
					}

					trialResult = {
						trialIdx: state.currentTrialIdx,
						cohensKappa: cohensKappa,
						matchResults: matchResults,
					};
				}

				state.currentTrialIdx++;
				if (state.currentTrialIdx > problemCount + n) {
					onUpdate?.(undefined, trialResult);
					reset();
					return;
				}

				let trial: Trial;
				trial = trialFactory.random();
				state.queue.push(trial);
				onUpdate?.(trial, trialResult);
			};

			loopTrial();
			timer = setInterval(() => {
				try {
					loopTrial();
				} catch (e) {
					reset();
					throw e;
				}
			}, interval);
			return reset;
		},
	};
};

export const newTrialFactory = ({
	stimulusTypes = [StimulusType.Character, StimulusType.Position],
	colors = Object.values(Color),
	shapes = Object.values(Shape),
	characters = [
		Character.ZERO,
		Character.ONE,
		Character.TWO,
		Character.THREE,
		Character.FOUR,
		Character.FIVE,
	],
	sounds = Object.values(Sound),
	animations = Object.values(Animation),
	gridSize = DefaultTrialFactoryOptions.gridSize,
}: TrialFactoryOptions): TrialFactory => {
	const random = (): Trial => {
		const stimuli: TrialStimuli = {};
		for (const type of stimulusTypes) {
			switch (type) {
				case StimulusType.Position:
					stimuli.position = [
						Math.floor(Math.random() * gridSize[0]),
						Math.floor(Math.random() * gridSize[1]),
					];
					break;
				case StimulusType.Color:
					stimuli.color = colors[Math.floor(Math.random() * colors.length)];
					break;
				case StimulusType.Character:
					stimuli.character =
						characters[Math.floor(Math.random() * characters.length)];
					break;
				case StimulusType.Shape:
					stimuli.shape = shapes[Math.floor(Math.random() * shapes.length)];
					break;
				case StimulusType.Sound:
					stimuli.sound = sounds[Math.floor(Math.random() * sounds.length)];
					break;
				case StimulusType.Animation:
					stimuli.animation =
						animations[Math.floor(Math.random() * animations.length)];
					break;
				default:
					throw new Error(`Unsupported stimulus type: ${type}`);
			}
		}

		return {
			compare(other: Trial): MatchResult[] {
				const results: MatchResult[] = [];
				for (const type of stimulusTypes) {
					let match = false;
					if (type === StimulusType.Position) {
						const value = stimuli[type];
						const otherValue = other.stimuli()[type];
						if (!value || !otherValue) {
							throw new Error("Position should be defined");
						}
						match = value[0] === otherValue[0] && value[1] === otherValue[1];
					} else {
						const value = stimuli[type];
						const otherValue = other.stimuli()[type];
						match = value === otherValue;
					}
					results.push({
						stimulusType: type,
						match,
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

const calculateCohensKappa = (
	hit: number,
	falseAlarm: number,
	miss: number,
	correctRejection: number,
): number => {
	const total = hit + falseAlarm + miss + correctRejection;
	if (total === 0) return Number.NaN; // 試行がない場合

	const p_o = (hit + correctRejection) / total;
	const p_e =
		((hit + falseAlarm) * (hit + miss) +
			(miss + correctRejection) * (falseAlarm + correctRejection)) /
		(total * total);

	// 分母が0の場合
	if (1 - p_e === 0) return Number.NaN;

	return (p_o - p_e) / (1 - p_e);
};
