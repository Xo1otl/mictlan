type TryResult = {
	reset(): void;
	matchResult: MatchResult;
};

function NewTryResult(): TryResult {
	const matchResult: MatchResult = {
		color: false,
		shape: false,
		coordinates: false,
		character: false,
	};
	function reset() {
		matchResult.color = false;
		matchResult.shape = false;
		matchResult.coordinates = false;
		matchResult.character = false;
	}
	return {
		reset,
		matchResult,
	};
}

interface Manager {
	start(): Promise<void>;
	input(tryResult: MatchResult): void;
	subscribe(listener: (event: string) => void): void;
}

export function NewManager(
	config: Config,
	scoreRecords: ScoreRecords,
	queue: Queue,
): Manager {
	let tryResult: TryResult = NewTryResult();
	let listeners: ((event: string) => void)[] = [];
	const input = (matchResult: MatchResult) => {
		tryResult.matchResult = matchResult;
	};
	const start = async () => {
		for (let i = 0; i < config.totalQuestions; i++) {
			const newStimulus = RandomStimulus(true, true, true, true);
			queue.push(newStimulus);
			console.log("Waiting for input");
			await new Promise((resolve) => setTimeout(resolve, config.delay));
			console.log("Input received: ", tryResult);

			if (queue.isFull()) {
				console.log("Stack is full");
				const target = queue.pop();
				console.log("Target: ", target);
				if (!target) {
					throw new Error("Target is missing");
				}
				const result = target.match(newStimulus);
				scoreRecords.update(tryResult.matchResult, result);
				tryResult.reset();
				for (const listener of listeners) {
					listener("update");
				}
			}
		}
	};
	const subscribe = (listener: (event: string) => void) => {
		listeners.push(listener);
	};
	return {
		start,
		input,
		subscribe,
	};
}

export type Config = {
	readonly n: number;
	readonly delay: number;
	readonly totalQuestions: number;
};

export type Score = {
	matches: number;
	total: number;
};

export interface ScoreRecords {
	coordinates: Score;
	character: Score;
	color: Score;
	shape: Score;
	total: Score;
	update: (tryResult: MatchResult, actualResult: MatchResult) => void;
}

interface Equatable<T> {
	equals(other: T): boolean;
}

const availableColors = ["red", "green", "blue", "yellow"] as const;
type Color = {
	value: string;
} & Equatable<Color>;
function NewColor(value: string): Color {
	if (!(availableColors as readonly string[]).includes(value)) {
		throw new Error("Invalid color");
	}
	const color: Color = {
		value: value,
		equals: (other) => {
			return color.value === other.value;
		},
	};
	return color;
}
function RandomColor(): Color {
	return NewColor(
		availableColors[Math.floor(Math.random() * availableColors.length)],
	);
}

type Charactor = {
	value: string;
} & Equatable<Charactor>;
function NewCharactor(value: string): Charactor {
	if (value.length !== 1) {
		throw new Error("Invalid character");
	}
	const character: Charactor = {
		value: value,
		equals: (other) => {
			return character.value === other.value;
		},
	};
	return character;
}
function RandomCharacter(): Charactor {
	const characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
	return NewCharactor(
		characters[Math.floor(Math.random() * characters.length)],
	);
}

type Coordinates = { x: number; y: number } & Equatable<Coordinates>;
function NewCoordinates(x: number, y: number): Coordinates {
	if (x < 0 || x >= 4 || y < 0 || y >= 4) {
		throw new Error("Invalid coordinates");
	}
	const coordinates: Coordinates = {
		x,
		y,
		equals: (other) => {
			return coordinates.x === other.x && coordinates.y === other.y;
		},
	};
	return coordinates;
}
function RandomCoordinates(): Coordinates {
	return NewCoordinates(
		Math.floor(Math.random() * 4),
		Math.floor(Math.random() * 4),
	);
}

const availableShapes = ["circle", "square", "triangle"];
type Shape = {
	value: string;
} & Equatable<Shape>;
function NewShape(value: string): Shape {
	if (!availableShapes.includes(value)) {
		throw new Error("Invalid shape");
	}
	const shape: Shape = {
		value: value,
		equals: (other) => {
			return shape.value === other.value;
		},
	};
	return shape;
}
function RandomShape(): Shape {
	return NewShape(
		availableShapes[Math.floor(Math.random() * availableShapes.length)],
	);
}

export type Stimulus = {
	color?: Color;
	shape?: Shape;
	coordinates?: Coordinates;
	character?: Charactor;
	match(target: Stimulus): MatchResult;
};

const RandomStimulus: (
	coordinates: boolean,
	character: boolean,
	color: boolean,
	shape: boolean,
) => Stimulus = (coordinates, character, color, shape) => {
	let stimulus: Stimulus = {
		match: (target: Stimulus) => {
			const result: MatchResult = {};
			if (color) {
				if (!stimulus.color || !target.color) {
					throw new Error("Color is missing");
				}
				result.color = stimulus.color.equals(target.color);
			}
			if (shape) {
				if (!stimulus.shape || !target.shape) {
					throw new Error("Shape is missing");
				}
				result.shape = stimulus.shape.equals(target.shape);
			}
			if (coordinates) {
				if (!stimulus.coordinates || !target.coordinates) {
					throw new Error("Coordinates are missing");
				}
				result.coordinates = stimulus.coordinates.equals(target.coordinates);
			}
			if (character) {
				if (!stimulus.character || !target.character) {
					throw new Error("Charactor is missing");
				}
				result.character = stimulus.character.equals(target.character);
			}
			return result;
		},
	};
	if (coordinates) {
		stimulus.coordinates = RandomCoordinates();
	}
	if (character) {
		stimulus.character = RandomCharacter();
	}
	if (color) {
		stimulus.color = RandomColor();
	}
	if (shape) {
		stimulus.shape = RandomShape();
	}
	return stimulus;
};

export type MatchResult = {
	[K in keyof Omit<Stimulus, "match">]: boolean;
};

export interface Queue {
	isFull(): boolean;
	push(stimulus: Stimulus): void;
	pop(): Stimulus | undefined;
}
