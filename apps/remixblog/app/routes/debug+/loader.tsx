import type { LoaderFunctionArgs } from "@remix-run/node";
import { useLoaderData } from "@remix-run/react";

export function loader({ request, context }: LoaderFunctionArgs): string {
	console.log("Loader function called");
	console.debug(context);
	return "AAAA";
}

export default function Loader() {
	const message = useLoaderData<typeof loader>();
	return (
		<div className="container mx-auto px-4 py-2">
			<h1 className="text-4xl font-bold">{message}</h1>
		</div>
	);
}
