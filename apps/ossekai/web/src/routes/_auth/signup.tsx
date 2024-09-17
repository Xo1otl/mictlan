import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import * as auth from "../../auth";
import * as utils from "lib/utilitytypes";
import { Navigate } from "@tanstack/react-router";
import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from "@/vendor/shadcn/components/ui/card";
import { Label } from "@/vendor/shadcn/components/ui/label";
import { Input } from "@/vendor/shadcn/components/ui/input";
import { Button } from "@/vendor/shadcn/components/ui/button";
import { Alert, AlertDescription } from "@/vendor/shadcn/components/ui/alert";

export function SignUp() {
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [confirmPassword, setConfirmPassword] = useState("");
	const [error, setError] = useState<string | null>(null);

	const [state, app] = auth.use();

	const handleSubmit = async (e: FormEvent) => {
		e.preventDefault();
		setError(null);

		if (password !== confirmPassword) {
			setError("パスワードが一致しません。");
			return;
		}

		try {
			const makePassword = utils.make<auth.Password>();
			await app.signUp(new auth.Username(username), makePassword(password));
		} catch (err) {
			setError(`サインアップに失敗しました。${err}`);
		}
	};

	if (state === "pendingConfirmation") {
		return <Navigate to="/confirm" />;
	}
	if (state !== "unauthenticated") {
		return <Navigate to="/home" />;
	}

	return (
		<div className="min-h-screen flex items-center justify-center bg-background">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>アカウント作成</CardTitle>
					<CardDescription>新しいアカウントを作成します</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-4">
						<div className="space-y-2">
							<Label htmlFor="username">ユーザー名</Label>
							<Input
								id="username"
								type="text"
								placeholder="ユーザー名を入力"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
								required
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="password">パスワード</Label>
							<Input
								id="password"
								type="password"
								placeholder="パスワードを入力"
								value={password}
								onChange={(e) => setPassword(e.target.value)}
								required
							/>
						</div>
						<div className="space-y-2">
							<Label htmlFor="confirm-password">パスワード（確認）</Label>
							<Input
								id="confirm-password"
								type="password"
								placeholder="パスワードを再入力"
								value={confirmPassword}
								onChange={(e) => setConfirmPassword(e.target.value)}
								required
							/>
						</div>
						{error && (
							<Alert variant="destructive">
								<AlertDescription>{error}</AlertDescription>
							</Alert>
						)}
						<Button className="w-full" type="submit">
							アカウント作成
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex justify-center">
					<Button variant="link" asChild>
						<Link to="/signin">既にアカウントをお持ちの方はこちら</Link>
					</Button>
				</CardFooter>
			</Card>
		</div>
	);
}

export const Route = createFileRoute("/_auth/signup")({
	component: () => <SignUp />,
});
