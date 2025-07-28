import re
from typing import Any, Dict, List

from mathruler.grader import extract_boxed_content, grade_answer

def conclusion2scores(_conclusion, _mapping):
    scores = {}        
    _conclusion = [item.strip() for item in _conclusion.split('\n') if item.strip()]
    for line in _conclusion:
        if len(scores) == len(_mapping):
            break
        line = line.strip()
        if not line:
            return 0

        # 去除反引号和其他markdown格式标记
        line = line.strip('`').strip()        
        
        # 查找第一个数字作为分数
        parts = line.split()
        if len(parts) < 3:  # 至少需要：维度名 分数 理由
            return 0
        dimension_name = parts[0]  # 中文维度名
        score = int(parts[1])  # 分数
        thinking = ' '.join(parts[2:])  # 理由部分
        
        # 转换为英文维度名
        english_name = _mapping[dimension_name]
        scores[english_name] = {
            'score': score,
            'thinking': thinking
        }
    return scores

_mapping = {
    '完备性': 'Sufficiency',
    '正确性': 'Correctness',
}

def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*正确性.*完备性.*", re.DOTALL)
    # <think>...</think>...正确性...完备性...
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0

def ectract_score(_str):
    _pre = _str.split('</think>') # for mimovl
    _pre_core = _pre[1]
    _conclusion = _pre_core.strip()
    _tmp_dict = conclusion2scores(_conclusion)
    score = 1.0
    for item in list(_mapping.values()):
        score *= _tmp_dict[item]['score']
    return score

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
