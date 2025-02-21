import type { FC } from "react";

type DrawerContentProps = {
	pageTree: string;
};

export const DrawerContent: FC<DrawerContentProps> = ({ pageTree }) => {
	return (
		<div className="">
			<div>{pageTree}</div>
			<div>AAA</div>
			<div>AAA</div>
			<div>AAA</div>
			<div>AAA</div>
			<div>AAA</div>
			<div>AAA</div>
		</div>
	);
};
