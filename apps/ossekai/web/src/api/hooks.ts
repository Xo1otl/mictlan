import * as auth from "../auth";

const API_BASE_URL = "http://localhost:3030";

interface ApiRequestConfig extends Omit<RequestInit, "body"> {
	path: string;
	body?: unknown;
}

export function useFetchJson() {
	const [state, app] = auth.use();

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

interface FormApiRequestConfig extends Omit<RequestInit, "body"> {
	path: string;
	body: FormData;
}

export function usePostForm() {
	const [state, app] = auth.use();

	return async ({ path, body, ...customConfig }: FormApiRequestConfig) => {
		const baseHeaders: Record<string, string> = {};
		if (state === "authenticated") {
			const token = await app.token();
			if (token) {
				baseHeaders.Authorization = `Bearer ${token}`;
			}
		}

		const url = `${API_BASE_URL}${path}`;
		const config: RequestInit = {
			method: "POST", // FormDataは通常POSTで送信されるため、デフォルトをPOSTに設定
			...customConfig,
			headers: {
				...baseHeaders,
				...customConfig.headers,
			},
		};

		config.body = body;

		try {
			const response = await fetch(url, config);
			if (response.ok) {
				return await response.json();
			}
			throw new Error(response.statusText);
		} catch (err) {
			return Promise.reject(err);
		}
	};
}
