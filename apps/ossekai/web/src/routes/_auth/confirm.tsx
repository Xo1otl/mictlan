import { createFileRoute } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { make } from "lib/utilitytypes";
import { Navigate } from "@tanstack/react-router";
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from "@/vendor/shadcn/components/ui/card";
import { Input } from "@/vendor/shadcn/components/ui/input";
import { Button } from "@/vendor/shadcn/components/ui/button";
import { Alert, AlertDescription } from "@/vendor/shadcn/components/ui/alert";
import { Label } from "@/vendor/shadcn/components/ui/label";
import * as auth from "@/src/auth";

export function Confirm() {
	const [code, setCode] = useState("");
	const [error, setError] = useState<string | null>(null);

	const [state, app] = auth.use();

	const handleSubmit = async (e: FormEvent) => {
		e.preventDefault();
		setError(null);

		try {
			const authCode = make<auth.Code>();
			await app.confirm(authCode(code));
		} catch (err) {
			setError(`確認に失敗しました。${err}`);
		}
	};

	if (state !== "pendingConfirmation") {
		return <Navigate to="/home" />;
	}

	return (
		<div className="min-h-screen flex items-center justify-center bg-background">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>アカウント確認</CardTitle>
					<CardDescription>確認コードを入力してください</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-4">
						<div className="space-y-2">
							<Label htmlFor="confirmation-code">確認コード</Label>
							<Input
								id="confirmation-code"
								type="text"
								placeholder="確認コードを入力"
								value={code}
								onChange={(e) => setCode(e.target.value)}
								required
							/>
						</div>
						{error && (
							<Alert variant="destructive">
								<AlertDescription>{error}</AlertDescription>
							</Alert>
						)}
						<Button className="w-full" type="submit">
							確認
						</Button>
					</form>
				</CardContent>
			</Card>
		</div>
	);
}

export const Route = createFileRoute("/_auth/confirm")({
	component: () => <Confirm />,
});
