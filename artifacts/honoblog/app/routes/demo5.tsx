import { createRoute } from "honox/factory";
import { Demo } from "../islands/Demo";

export default createRoute((c) => {
	return c.render(<Demo />);
});
