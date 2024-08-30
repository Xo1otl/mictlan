import { setup } from "xstate";

export const machine = setup({
  types: {
    context: {} as object,
    events: {} as
      | { type: "signIn" }
      | { type: "signUp" }
      | { type: "signOut" }
      | { type: "confirm" },
  },
}).createMachine({
  /** @xstate-layout N4IgpgJg5mDOIC5QEMCuAXAFmAduglgMbIED2OAdKjmlrgcepAMSz5Q4CqADgNoAMAXUShupNmRwiQAD0QAmACz8KATgBsADgDsi1fICMqxfICs8gMwAaEAE9EBiwbWrX2-qs0Gl6gwF8-G1psPCISfHIqGgwQhhIWNg4ASRwBYSQQMQkIqQy5BCUVDR09Q2MzSxt7Aot5F1dFI3UlfXl5AKCY+jDJCmDuxgT2HAB5DDTpLPxJaXzCtS1dfSMTc2s7B006vTcPLx9-QJB+0MYcim5cCHwcKABhcgAzfAAnAFtw8mZCJ9e3iYyUxmeQUJgoBm0Bh0-AsMK2FmaVUQXgopgCRxwpAgcGkJziwNE4mmOVmiAAtOokQgyaYKFooaZGppTNp1Kp+KY0Uc8T1ztQeYMIJMiQTZIgTFTVBYKPwdtp5HD5LpzB1jl1Tp9KAL4kLASKSSCEAjnB4ETpDAZ+OoOdoqUYTZpauptBZNIpmgZ1Krtb1LjhrrcHjhnu9NcLsuRSUbfDKpYsLVabZKVPxHbU3cVdBZFOi-EA */
  context: {},
  id: "authentication",
  initial: "unauthenticated",
  states: {
    unauthenticated: {
      on: {
        signUp: {
          target: "pendingConfirmation",
          reenter: true,
        },

        signIn: {
          target: "authenticated",
          reenter: true,
        },
      },
    },

    authenticated: {
      on: {
        signOut: {
          target: "unauthenticated",
          reenter: true,
        },
      },
    },

    pendingConfirmation: {
      on: {
        confirm: {
          target: "unauthenticated",
          reenter: true,
        },
      },
    },
  },
});
