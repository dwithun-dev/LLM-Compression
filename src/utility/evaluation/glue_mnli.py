"""GLUE Multi-Genre Natural Language Inference (MNLI) Evaluation Suite."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm


DEFAULT_LABELS = ["entailment", "neutral", "contradiction"]

DEFAULT_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "Premise: {premise}\n"
    "Hypothesis: {hypothesis}\n"
    "Determine if the relationship between the 'Premise' and 'Hypothesis' is 'entailment', 'neutral' or 'contradiction.'\n"
    "Answer with one word\n"
    "<start_of_turn>model\n"
)


def evaluate_mnli(
    model,
    tokenizer,
    dataset_split,
    sample_limit: Optional[int] = None,
    label_names: Optional[List[str]] = None,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    step_callback: Optional[Callable[[], None]] = None,
    desc: str = "Evaluating MNLI",
) -> Dict[str, object]:
    """Run standardized GLUE MNLI evaluation on a causal language model.

    Args:
        model: HuggingFace causal language model.
        tokenizer: HuggingFace tokenizer.
        dataset_split: Dataset split iterable (e.g. ds['validation_matched']).
        sample_limit: Optional subset length to evaluate.
        label_names: Target label strings (default: entailment, neutral, contradiction).
        prompt_template: String template containing {premise} and {hypothesis}.
        step_callback: Optional callback invoked after each model forward step (e.g. for hook pooling).
        desc: Progress bar description.

    Returns:
        Dictionary containing accuracy, classification_report string, predictions, and ground_truth.
    """
    labels = label_names or DEFAULT_LABELS
    label_token_ids = [tokenizer.encode(" " + name, add_special_tokens=False)[0] for name in labels]

    eval_data = dataset_split.select(range(sample_limit)) if sample_limit else dataset_split

    predictions: List[int] = []
    ground_truth: List[int] = []

    model.eval()
    with torch.no_grad():
        for sample in tqdm(eval_data, desc=desc):
            prompt = prompt_template.format(
                premise=sample["premise"],
                hypothesis=sample["hypothesis"],
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs)

            if step_callback is not None:
                step_callback()

            next_token_logits = outputs.logits[0, -1, :]
            candidate_logits = next_token_logits[label_token_ids]
            pred_label = torch.argmax(candidate_logits).item()

            predictions.append(pred_label)
            ground_truth.append(sample["label"])

    accuracy = float(accuracy_score(ground_truth, predictions))
    report_str = classification_report(
        ground_truth,
        predictions,
        labels=[0, 1, 2],
        target_names=labels,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "classification_report": report_str,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }
