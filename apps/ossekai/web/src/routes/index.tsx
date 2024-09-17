import { createFileRoute, Link } from "@tanstack/react-router";

export const Route = createFileRoute("/")({
	component: () => <Link to="/home">Home</Link>,
});
