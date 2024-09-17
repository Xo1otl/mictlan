import { useReducer } from "react";
import * as api from "@/src/api";

export const useForm = () => {
	const [state, dispatch] = useReducer(formReducer, {
		isValid: false,
		title: "",
		tagNames: [],
		contentBlocks: [{ id: crypto.randomUUID(), kind: "text", content: "" }],
		files: [],
	});

	const postForm = api.usePostForm();

	const handleSubmit = async (e: { preventDefault: () => void }) => {
		e.preventDefault();
		const formData = new FormData();

		formData.append("title", state.title);
		for (const item of state.tagNames) {
			formData.append("tag_names", item.tagName);
		}

		state.contentBlocks.forEach((block, index) => {
			formData.append(`contentBlocks[${index}][kind]`, block.kind);
			formData.append(`contentBlocks[${index}][content]`, block.content);
		});

		for (const item of state.files) {
			formData.append("files", item.file);
		}

		try {
			const result = await postForm({
				path: "/qa/ask-question",
				body: formData,
			});
			console.log(result);
		} catch (error) {
			console.error("Error submitting form:", error);
		}
	};

	return {
		state,
		dispatch,
		handleSubmit,
	};
};

export interface FormState {
	isValid: boolean;
	title: string;
	tagNames: { id: string; tagName: string }[];
	contentBlocks: {
		id: string;
		kind: "text" | "latex" | "markdown";
		content: string;
	}[];
	files: { id: string; file: File }[];
}

export type FormAction =
	| { type: "SET_TITLE"; payload: { title: string } }
	| { type: "ADD_TAG"; payload: { id: string; tagName: string } }
	| { type: "REMOVE_TAG"; payload: { id: string } }
	| {
			type: "UPDATE_TAG";
			payload: { index: number; value: { id: string; tagName: string } };
	  }
	| { type: "ADD_CONTENT_BLOCK" }
	| { type: "REMOVE_CONTENT_BLOCK"; payload: { id: string } }
	| {
			type: "UPDATE_CONTENT_BLOCK";
			payload: { id: string; field: "kind" | "content"; value: string };
	  }
	| { type: "ADD_FILE"; payload: { file: File } }
	| { type: "REMOVE_FILE"; payload: { id: string } };

const formReducer = (state: FormState, action: FormAction): FormState => {
	let newState: FormState;
	switch (action.type) {
		case "SET_TITLE":
			newState = { ...state, title: action.payload.title };
			break;
		case "ADD_TAG":
			newState = {
				...state,
				tagNames: [
					...state.tagNames,
					{ id: action.payload.id, tagName: action.payload.tagName },
				],
			};
			break;
		case "REMOVE_TAG":
			newState = {
				...state,
				tagNames: state.tagNames.filter((tag) => tag.id !== action.payload.id),
			};
			break;
		case "UPDATE_TAG":
			newState = {
				...state,
				tagNames: state.tagNames.map((tag, index) =>
					index === action.payload.index ? action.payload.value : tag,
				),
			};
			break;
		case "ADD_CONTENT_BLOCK":
			newState = {
				...state,
				contentBlocks: [
					...state.contentBlocks,
					{ id: crypto.randomUUID(), kind: "text", content: "" },
				],
			};
			break;
		case "REMOVE_CONTENT_BLOCK":
			newState = {
				...state,
				contentBlocks: state.contentBlocks.filter(
					(block) => block.id !== action.payload.id,
				),
			};
			break;
		case "UPDATE_CONTENT_BLOCK":
			newState = {
				...state,
				contentBlocks: state.contentBlocks.map((block) =>
					block.id === action.payload.id
						? { ...block, [action.payload.field]: action.payload.value }
						: block,
				),
			};
			break;
		case "ADD_FILE":
			newState = {
				...state,
				files: [
					...state.files,
					{ id: crypto.randomUUID(), file: action.payload.file },
				],
			};
			break;
		case "REMOVE_FILE":
			newState = {
				...state,
				files: state.files.filter((file) => file.id !== action.payload.id),
			};
			break;
	}

	// 一旦falseにしてから検証開始
	newState.isValid = false;
	// ファイルがblockに含まれているかなどのvalidationも追加したいけど、バックエンドではすべて行ってるからめんどいなら後回し
	if (!newState.title || newState.contentBlocks.length === 0) {
		return newState;
	}
	// early returnでformがvalidか全部の項目をチェック
	for (const block of newState.contentBlocks) {
		if (!block.content) {
			return newState;
		}
	}
	// 全部通ったらvalid
	newState.isValid = true;
	return newState;
};
