import { createFileRoute } from "@tanstack/react-router";
import { useAuth, useAuthenticatedFetch } from "@/internal/react/hooks/auth";
import { Button } from "@/vendor/shadcn/components/ui/button";
import {
	Card,
	CardHeader,
	CardTitle,
	CardContent,
} from "@/vendor/shadcn/components/ui/card";
import { useState } from "react";

export const Route = createFileRoute(
	"/_auth/_protected/_layout/debug",
)({
	component: () => {
		const [, app] = useAuth();
		const authenticatedFetch = useAuthenticatedFetch();
		const [pressedButton, setPressedButton] = useState<string | null>(null);
		const [response, setResponse] = useState<string | null>(null);

		const handleButtonClick = async (buttonName: string) => {
			setPressedButton(buttonName);

			if (buttonName === "ShowUser") {
				try {
					setResponse(`${(await app.user())?.username}`);
				} catch (error) {
					console.error("Error:", error);
					setResponse(`Error: ${error}`);
				}
			}
			if (buttonName === "ShowToken") {
				try {
					setResponse(`${await app.token()}`);
				} catch (error) {
					console.error("Error:", error);
					setResponse(`Error: ${error}`);
				}
			}
			if (buttonName === "Fetch") {
				setResponse("Fetching data...");
				try {
					const response = await authenticatedFetch(
						"http://localhost:3000/qa/answers",
						{
							method: "POST",
							headers: {
								"Content-Type": "application/json",
							},
							body: JSON.stringify({
								// ここに送信したいデータを追加します
								// 例: question: 'What is the meaning of life?'
							}),
						},
					);

					if (!response.ok) {
						throw new Error(`HTTP error! status: ${response.status}`);
					}

					const data = await response.json();
					setResponse(JSON.stringify(data, null, 2));
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
						<Button onClick={() => handleButtonClick("ShowUser")}>
							ShowUser
						</Button>
						<Button onClick={() => handleButtonClick("ShowToken")}>
							ShowToken
						</Button>
						<Button onClick={() => handleButtonClick("Fetch")}>Fetch</Button>
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
