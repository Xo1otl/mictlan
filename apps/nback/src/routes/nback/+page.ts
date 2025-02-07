// since there's no dynamic data here, we can prerender
// it so that it gets served as a static asset in production
import type * as nback from "../../nback/index";

export const prerender = true;

export type Config = {
	trialFactoryOptions: nback.TrialFactoryOptions;
	taskEngineOptions: Omit<nback.TaskEngineOptions, "trialFactory">;
	answerDisplayTime: number;
};
