import { useState, type FC } from "hono/jsx";

export const Counter: FC = () => {
	const [count, setCount] = useState(0);
	return (
		<div>
			<div>{count}</div>
			{/* <div>AAA</div> */}
			<button type="button" onClick={() => setCount(count + 1)}>
				Increment
			</button>
		</div>
	);
};
