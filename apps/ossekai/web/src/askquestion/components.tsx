import React from "react";
import { Input } from "@/vendor/shadcn/components/ui/input";
import { Button } from "@/vendor/shadcn/components/ui/button";
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
import { type FormState, useForm, type FormAction } from ".";
import { Plus, X } from "lucide-react";
import { AutosizeTextarea } from "../../vendor/shadcn/components/ui/autosize-textarea";

const TagSection = ({
	tagNames,
	dispatch,
}: {
	tagNames: FormState["tagNames"];
	dispatch: React.Dispatch<FormAction>;
}) => {
	const inputRef = React.useRef<HTMLInputElement>(null);

	const handleAddTag = (tagValue: string) => {
		// TODO: apiを使用して存在するタグかどうか検証する
		// 既存のタグがある場合、そのidを使用する
		const trimmedTag = tagValue.trim();
		if (trimmedTag !== "") {
			dispatch({
				type: "ADD_TAG",
				payload: { id: crypto.randomUUID(), tagName: trimmedTag },
			});
		}
	};

	const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
		const { key } = e;
		const target = e.target as HTMLInputElement;
		const value = target.value;

		if (key === "Enter" || key === "Tab" || key === ",") {
			e.preventDefault();
			handleAddTag(value);
			target.value = "";
		} else if (key === "Backspace" && value === "") {
			e.preventDefault();
			if (tagNames.length > 0) {
				dispatch({
					type: "REMOVE_TAG",
					payload: { id: tagNames[tagNames.length - 1].id },
				});
			}
		}
	};

	return (
		<div className="flex items-center">
			<div
				className="flex items-center flex-wrap overflow-x-auto hide-scrollbar py-1"
				style={{ maxWidth: "100%" }}
			>
				{tagNames.map((item) => (
					<div
						key={item.id}
						className="flex items-center bg-gray-200 rounded-full px-2 py-1 mr-1 mb-1 text-sm"
					>
						<span className="mr-1">{item.tagName}</span>
						<button
							type="button"
							onClick={() =>
								dispatch({ type: "REMOVE_TAG", payload: { id: item.id } })
							}
							className="text-gray-500 hover:text-gray-700 focus:outline-none"
						>
							<X size={14} />
						</button>
					</div>
				))}
				<input
					type="text"
					ref={inputRef}
					onKeyDown={handleInputKeyDown}
					placeholder="Add tags"
					className="border-none outline-none text-sm flex-grow min-w-[100px] py-1"
					style={{ flexBasis: "100px" }}
				/>
			</div>
		</div>
	);
};

const ContentBlockSection = ({
	contentBlocks,
	dispatch,
}: {
	contentBlocks: FormState["contentBlocks"];
	dispatch: React.Dispatch<FormAction>;
}) => (
	<div className="relative pb-3">
		<div className="space-y-4">
			{contentBlocks.map((block, index) => (
				<div key={block.id} className="relative">
					<div className="absolute bottom-0 right-0 z-10">
						<Select
							value={block.kind}
							onValueChange={(value) => {
								if (value === "delete") {
									dispatch({
										type: "REMOVE_CONTENT_BLOCK",
										payload: { id: block.id },
									});
									return;
								}
								dispatch({
									type: "UPDATE_CONTENT_BLOCK",
									payload: { id: block.id, field: "kind", value },
								});
							}}
						>
							<SelectTrigger className="h-6 border-none text-xs bg-transparent focus:ring-0 focus:ring-offset-0">
								<SelectValue placeholder="Type" />
							</SelectTrigger>
							<SelectContent>
								<SelectItem value="text">Text</SelectItem>
								<SelectItem value="latex">LaTeX</SelectItem>
								<SelectItem value="markdown">Markdown</SelectItem>
								<SelectItem value="delete">Delete</SelectItem>
							</SelectContent>
						</Select>
					</div>
					<AutosizeTextarea
						value={block.content}
						onChange={(e) =>
							dispatch({
								type: "UPDATE_CONTENT_BLOCK",
								payload: {
									id: block.id,
									field: "content",
									value: e.target.value,
								},
							})
						}
						placeholder={`Content for block ${index + 1}`}
						className="w-full pb-5"
					/>
				</div>
			))}
		</div>
		<Button
			type="button"
			onClick={() => dispatch({ type: "ADD_CONTENT_BLOCK" })}
			variant="outline"
			className="absolute left-1/2 bottom-0 -translate-x-1/2 w-6 h-6 p-0 rounded-full"
			title="Add Content Block"
		>
			<Plus className="w-4 h-4" />
		</Button>
	</div>
);

const FileSection = ({
	files,
	dispatch,
}: { files: FormState["files"]; dispatch: React.Dispatch<FormAction> }) => (
	<div className="space-y-2">
		{files.map((fileObj) => (
			<div key={fileObj.id} className="flex items-center space-x-2">
				<span>{fileObj.file.name}</span>
				<Button
					type="button"
					onClick={() =>
						dispatch({ type: "REMOVE_FILE", payload: { id: fileObj.id } })
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
						payload: { file: e.target.files[0] },
					});
				}
			}}
			className="w-full"
		/>
	</div>
);

export const Form = () => {
	const { state, dispatch, handleSubmit } = useForm();

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
								dispatch({
									type: "SET_TITLE",
									payload: { title: e.target.value },
								})
							}
							placeholder="Title"
							className="w-full"
						/>
					</div>

					<TagSection tagNames={state.tagNames} dispatch={dispatch} />

					<ContentBlockSection
						contentBlocks={state.contentBlocks}
						dispatch={dispatch}
					/>

					<FileSection files={state.files} dispatch={dispatch} />

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
