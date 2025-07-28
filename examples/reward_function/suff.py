import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*完备性.*", re.DOTALL)
    # <think>...</think>完备性...
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def ectract_score(_str):
    if '0' in _str:
        return 0
    else:
        return 1
    
def accuracy_reward(response: str, ground_truth: str) -> float:
    try:
        score = ectract_score(response)
    except:
        return 0.0
    return 1.0 if score == int(ground_truth) else 0.0


def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
