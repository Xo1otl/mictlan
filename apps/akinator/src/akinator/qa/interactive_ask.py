from . import Context, Selector
import torch


def check_status(context: Context, top_n: int = 3):
    top_probs, top_indices = torch.topk(
        context.p_case_tensor, min(top_n, len(context.p_case_tensor)))

    print("Current most likely cases:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        print(
            f"  {i+1}. {context.case_idx_to_id[int(idx.item())]} ({prob.item():.4f})")

    if top_probs[0].item() > 0.5:
        print(
            f"The most likely case is {context.case_idx_to_id[int(top_indices[0].item())]} with probability {top_probs[0].item():.4f}.")
        exit(1)


def interactive_ask(context: Context):
    selector = Selector(context)

    while True:
        best_question_id = selector.best_question()

        if best_question_id is None:
            raise Exception("Best question is None.")

        print(f"Question: {best_question_id}?")

        choice_labels = list(context.choice_to_idx.keys())
        for i, label in enumerate(choice_labels):
            print(f"{label}: {i+1}, ", end="")
        print()

        while True:
            try:
                choice_idx = int(input(f"Answer (1-{len(choice_labels)}): "))
                if 1 <= choice_idx <= len(choice_labels):
                    choice = choice_labels[choice_idx-1]
                    break
                else:
                    print(
                        f"Invalid choice. Please enter a number between 1 and {len(choice_labels)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

        selector.update_context(best_question_id, choice)
        check_status(context, 5)
