import { useState, type FC } from "hono/jsx";

export const Demo: FC = ({ children }) => {
	const [count, setCount] = useState(0);
	return (
		<>
			<div>{count}</div>
			<button
				type="button"
				onClick={() => {
					console.log("clicked");
					setCount(count + 1);
				}}
			>
				Increment
			</button>
			{children}
		</>
	);
};
