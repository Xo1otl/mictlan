import { createFileRoute, Link } from "@tanstack/react-router";
import { useState, type FormEvent } from "react";
import { useAuth } from "../../hooks/useAuth";
import * as auth from "../../../auth";
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

export function SignIn() {
	const [username, setUsername] = useState("");
	const [password, setPassword] = useState("");
	const [error, setError] = useState<string | null>(null);
	const [state, app] = useAuth();

	const handleSubmit = async (e: FormEvent): Promise<void> => {
		e.preventDefault();
		setError(null);

		try {
			const makePassword = utils.make<auth.Password>();
			await app.signIn(new auth.Username(username), makePassword(password));
		} catch (err) {
			setError(
				`サインインに失敗しました。ユーザー名とパスワードを確認してください。${err}`,
			);
		}
	};

	if (state !== "unauthenticated") {
		return <Navigate to="/home" />;
	}

	return (
		<div className="min-h-screen flex items-center justify-center bg-background">
			<Card className="w-full max-w-sm">
				<CardHeader>
					<CardTitle>サインイン</CardTitle>
					<CardDescription>アカウントにサインインしてください</CardDescription>
				</CardHeader>
				<CardContent>
					<form onSubmit={handleSubmit} className="space-y-4">
						<div className="space-y-2">
							<Label htmlFor="username">ユーザー名</Label>
							<Input
								id="username"
								placeholder="ユーザー名を入力"
								value={username}
								onChange={(e) => setUsername(e.target.value)}
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
							/>
						</div>
						{error && (
							<Alert variant="destructive">
								<AlertDescription>{error}</AlertDescription>
							</Alert>
						)}
						<Button className="w-full" type="submit">
							サインイン
						</Button>
					</form>
				</CardContent>
				<CardFooter className="flex justify-center">
					<Button variant="link" asChild>
						<Link to="/signup">アカウントをお持ちでない方はこちら</Link>
					</Button>
				</CardFooter>
			</Card>
		</div>
	);
}

export const Route = createFileRoute("/_auth/signin")({
	component: () => <SignIn />,
});
