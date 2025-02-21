import { useState, type FC } from "hono/jsx";

export const Demo: FC = ({ children }) => {
	const [count, setCount] = useState(0);
	return (
		<>
			<div>{count}</div>
			<button
				type="button"
				className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
				onClick={() => {
					console.log("clicked");
					setCount(count + 1);
				}}
			>
				Increment
			</button>
			<div className="prose lg:prose-xl max-w-none">{children}</div>
		</>
	);
};
