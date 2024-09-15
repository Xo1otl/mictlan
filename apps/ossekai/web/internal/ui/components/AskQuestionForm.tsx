import { useReducer } from "react";
import { useFormApi } from "../hooks/useApi";
import { Input } from "@/vendor/shadcn/components/ui/input";
import { Button } from "@/vendor/shadcn/components/ui/button";
import { Textarea } from "@/vendor/shadcn/components/ui/textarea";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/vendor/shadcn/components/ui/select";
import {
	Card,
	CardHeader,
	CardTitle,
	CardContent,
	CardFooter,
} from "@/vendor/shadcn/components/ui/card";

interface FormState {
	isValid: boolean;
	title: string;
	tagIds: { id: string; tagId: string }[];
	contentBlocks: {
		id: string;
		kind: "text" | "latex" | "markdown";
		content: string;
	}[];
	files: { id: string; file: File }[];
}

// TODO: content blockの削除機能が必要
type FormAction =
	| { type: "SET_TITLE"; payload: string }
	| { type: "ADD_TAG" }
	| {
			type: "UPDATE_TAG";
			payload: { index: number; value: { id: string; tagId: string } };
	  }
	| { type: "ADD_CONTENT_BLOCK" }
	| {
			type: "UPDATE_CONTENT_BLOCK";
			payload: { index: number; field: "kind" | "content"; value: string };
	  }
	| { type: "ADD_FILE"; payload: File }
	| { type: "REMOVE_FILE"; payload: string };

const formReducer = (state: FormState, action: FormAction): FormState => {
	let newState: FormState;
	switch (action.type) {
		case "SET_TITLE":
			newState = { ...state, title: action.payload };
			break;
		case "ADD_TAG":
			newState = {
				...state,
				tagIds: [...state.tagIds, { id: crypto.randomUUID(), tagId: "" }],
			};
			break;
		case "UPDATE_TAG":
			newState = {
				...state,
				tagIds: state.tagIds.map((tag, index) =>
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
		case "UPDATE_CONTENT_BLOCK":
			newState = {
				...state,
				contentBlocks: state.contentBlocks.map((block, index) =>
					index === action.payload.index
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
					{ id: crypto.randomUUID(), file: action.payload },
				],
			};
			break;
		case "REMOVE_FILE":
			newState = {
				...state,
				files: state.files.filter((file) => file.id !== action.payload),
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

export const AskQuestionForm = () => {
	const [state, dispatch] = useReducer(formReducer, {
		isValid: false,
		title: "",
		tagIds: [],
		contentBlocks: [],
		files: [],
	});

	const sendFormData = useFormApi();

	const handleSubmit = async (e: { preventDefault: () => void }) => {
		e.preventDefault();
		const formData = new FormData();

		formData.append("title", state.title);
		for (const item of state.tagIds) {
			formData.append("tag_ids", item.tagId);
		}

		state.contentBlocks.forEach((block, index) => {
			formData.append(`contentBlocks[${index}][kind]`, block.kind);
			formData.append(`contentBlocks[${index}][content]`, block.content);
		});

		for (const item of state.files) {
			formData.append("files", item.file);
		}

		try {
			const result = await sendFormData({
				path: "/qa/ask-question",
				body: formData,
			});
			console.log(result);
		} catch (error) {
			console.error("Error submitting form:", error);
		}
	};

	return (
		<Card className="w-full max-w-2xl mx-auto">
			<CardHeader>
				<CardTitle>Ask a Question</CardTitle>
			</CardHeader>
			<CardContent>
				<form onSubmit={handleSubmit} className="space-y-4">
					<div>
						<Input
							type="text"
							value={state.title}
							onChange={(e) =>
								dispatch({ type: "SET_TITLE", payload: e.target.value })
							}
							placeholder="Title"
							className="w-full"
						/>
					</div>

					<div className="space-y-2">
						{state.tagIds.map((item, index) => (
							<Input
								key={item.id}
								type="text"
								value={item.tagId}
								onChange={(e) =>
									dispatch({
										type: "UPDATE_TAG",
										payload: {
											index,
											value: { id: item.id, tagId: e.target.value },
										},
									})
								}
								placeholder={`Tag ${index + 1}`}
								className="w-full"
							/>
						))}
						<Button
							type="button"
							onClick={() => dispatch({ type: "ADD_TAG" })}
							variant="outline"
							className="w-full"
						>
							Add Tag
						</Button>
					</div>

					<div className="space-y-4">
						{state.contentBlocks.map((block, index) => (
							<div key={block.id} className="space-y-2">
								<Select
									value={block.kind}
									onValueChange={(value) =>
										dispatch({
											type: "UPDATE_CONTENT_BLOCK",
											payload: { index, field: "kind", value },
										})
									}
								>
									<SelectTrigger>
										<SelectValue placeholder="Select content type" />
									</SelectTrigger>
									<SelectContent>
										<SelectItem value="text">Text</SelectItem>
										<SelectItem value="latex">LaTeX</SelectItem>
										<SelectItem value="markdown">Markdown</SelectItem>
									</SelectContent>
								</Select>
								<Textarea
									value={block.content}
									onChange={(e) =>
										dispatch({
											type: "UPDATE_CONTENT_BLOCK",
											payload: {
												index,
												field: "content",
												value: e.target.value,
											},
										})
									}
									placeholder={`Content for block ${index + 1}`}
									className="w-full"
								/>
							</div>
						))}
						<Button
							type="button"
							onClick={() => dispatch({ type: "ADD_CONTENT_BLOCK" })}
							variant="outline"
							className="w-full"
						>
							Add Content Block
						</Button>
					</div>

					{state.files.map((fileObj) => (
						<div key={fileObj.id} className="flex items-center space-x-2">
							<span>{fileObj.file.name}</span>
							<Button
								type="button"
								onClick={() =>
									dispatch({ type: "REMOVE_FILE", payload: fileObj.id })
								}
								variant="outline"
								size="sm"
							>
								Remove
							</Button>
						</div>
					))}
					<Input
						type="file"
						onChange={(e) => {
							if (e.target.files?.[0]) {
								dispatch({
									type: "ADD_FILE",
									payload: e.target.files[0],
								});
							}
						}}
						className="w-full"
					/>

					<CardFooter className="px-0">
						<Button type="submit" className="w-full" disabled={!state.isValid}>
							Submit
						</Button>
					</CardFooter>
				</form>
			</CardContent>
		</Card>
	);
};
