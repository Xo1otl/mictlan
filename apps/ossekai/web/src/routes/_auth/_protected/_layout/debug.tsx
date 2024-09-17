import { createFileRoute } from "@tanstack/react-router";
import { Button } from "@/vendor/shadcn/components/ui/button";
import {
	Card,
	CardHeader,
	CardTitle,
	CardContent,
} from "@/vendor/shadcn/components/ui/card";
import { useState } from "react";
import * as api from "@/src/api";
import * as askquestion from "@/src/askquestion";

export const Route = createFileRoute("/_auth/_protected/_layout/debug")({
	component: () => {
		const [pressedButton, setPressedButton] = useState<string | null>(null);
		const [response, setResponse] = useState<string | null>(null);
		const fetchJson = api.useFetchJson();

		const handleButtonClick = async (buttonName: string) => {
			setPressedButton(buttonName);

			if (buttonName === "Fetch") {
				setResponse("Fetching data...");
				try {
					const response = await fetchJson({
						method: "POST",
						path: "/qa/answers",
						body: { question: "What is the meaning of life?" },
					});
					setResponse(JSON.stringify(response, null, 2));
				} catch (error) {
					console.error("Error:", error);
					setResponse(`Error: ${error}`);
				}
			}
		};

		return (
			<Card>
				<CardHeader>
					<CardTitle>Debug Component</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="flex space-x-2 mb-4">
						<Button onClick={() => handleButtonClick("Fetch")}>Fetch</Button>
					</div>
					<div className="flex space-x-2 mb-4">
						<askquestion.Form />
					</div>
					{pressedButton && <p>Last pressed button: {pressedButton}</p>}
					{response && (
						<div className="mt-4">
							<h3 className="font-semibold">Response:</h3>
							<pre className="bg-muted p-2 rounded mt-2 overflow-x-auto">
								{response}
							</pre>
						</div>
					)}
				</CardContent>
			</Card>
		);
	},
});
