import type React from "react";
import { useState } from "react";
import { Button } from "@/components/ui/button";

export const Counter: React.FC = () => {
	const [count, setCount] = useState(0);

	return (
		<Button onClick={() => setCount((count) => count + 1)}>
			Counter {count}
		</Button>
	);
};
