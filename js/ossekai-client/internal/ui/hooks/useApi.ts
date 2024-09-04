import { useAuth } from "./useAuth";

const API_BASE_URL = "http://localhost:3000";

interface ApiRequestConfig extends Omit<RequestInit, "body"> {
	path: string;
	body: unknown;
}

export function useApi() {
	const [state, app] = useAuth();

	return async ({ path, body, ...customConfig }: ApiRequestConfig) => {
		const baseHeaders: Record<string, string> = {
			"Content-Type": "application/json",
		};
		if (state === "authenticated") {
			const token = await app.token();
			if (token) {
				baseHeaders.Authorization = `Bearer ${token}`;
			}
		}
		const url = `${API_BASE_URL}${path}`;
		const config: RequestInit = {
			...customConfig,
			headers: {
				...baseHeaders,
				...customConfig.headers,
			},
		};

		// Only add body if it exists
		if (body !== undefined) {
			config.body = JSON.stringify(body);
		}

		try {
			const respose = await fetch(url, config);
			if (respose.ok) {
				return await respose.json();
			}
			throw new Error(respose.statusText);
		} catch (err) {
			return Promise.reject(err);
		}
	};
}
