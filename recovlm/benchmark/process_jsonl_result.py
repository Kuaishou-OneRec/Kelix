import json
import random
import re
from typing import Dict, List, Tuple

def extract_predicted_label(response: str) -> str:
    """Extract predicted label from model response"""
    # Look for label after ### marker
    if '###' in response:
        after_marker = response.split('###')[-1].strip()
        satisfaction_patterns = [
            r"【结果[:：]满意】",
            r"【结果[:：]不满意】",
            r"【結果：满意】",
            r"【結果：不满意】",
            r"【满意】",
            r"【不满意】",
            r"\*\*结果\*\*[:：]满意",
            r"\*\*结果\*\*[:：]不满意",
            r"结果[:：]\s*满意",
            r"结果[:：]\s*不满意",
            r"\*\*不满意\*\*",
            r"\*\*满意\*\*",
            r"\*\* 不满意 \*\*",
            r"\*\* 满意 \*\*",
            r"\*\*  不满意  \*\*",
            r"\*\*  满意  \*\*",
            r"结果：满意",
            r"结果：不满意",
        ]
        
        # 遍历所有可能的模式
        for pattern in satisfaction_patterns:
            match = re.search(pattern, after_marker)
            if match:
                return "不满意" if "不满意" in match.group() else "满意"
            elif "不满意" in after_marker:
                return "不满意"
            elif "满意" in after_marker:
                return "满意"
            
        return None
    elif '结论' in response:
        after_marker = response.split('结论')[-1].strip()
        if "不满意" in after_marker:
            return "不满意"
        elif "满意" in after_marker:
            return "满意"
    elif '【结果' in response:
        after_marker = response.split('【结果')[-1].strip()
        if "不满意" in after_marker:
            return "不满意"
        elif "满意" in after_marker:
            return "满意"
    elif '结果' in response:
        after_marker = response.split('结果')[-1].strip()
        if "不满意" in after_marker:
            return "不满意"
        elif "满意" in after_marker:
            return "满意"
    return None

def calculate_metrics(results: List[Dict]) -> Tuple[Dict[str, int], float]:
    """Calculate confusion matrix metrics and accuracy"""
    confusion_matrix = {
        'TP': 0,  # True Positive (预测满意，实际满意)
        'TN': 0,  # True Negative (预测不满意，实际不满意)
        'FP': 0,  # False Positive (预测满意，实际不满意)
        'FN': 0,   # False Negative (预测不满意，实际满意)
        "true_label_not_valid": 0,
        "pred_label_not_valid": 0,
        "no_pred_response": 0,
        "final_match_not_valid": 0
    }
    
    total = 0
    correct = 0
    
    for item in results:
        true_label = item['true_label']
        if true_label is None:
            confusion_matrix["true_label_not_valid"] += 1
            print(f"Warning: True label is None for item: {item}")
            continue
        else:
            true_label = extract_predicted_label(true_label)
        if true_label == "满意":
            # 满意样本中，有57%的样本会被过滤掉
            reject_random = random.random() < 0.57
            if reject_random:
                continue
        pred_label = None
        # Extract prediction from response
        if 'rsp' in item and item['rsp']:
            response = item['rsp'][0]['response']
            pred_label = extract_predicted_label(response)
            if pred_label is None:
                print(f"Warning: Predicted label is None for response: {response}")
                confusion_matrix["pred_label_not_valid"] += 1
                continue
        else:
            confusion_matrix["no_pred_response"] += 1
            print(f"Warning: Response is None for item: {item}")
            continue
        if pred_label is not None:
            total += 1
            if true_label == pred_label:
                correct += 1
                if true_label == '满意':
                    confusion_matrix['TP'] += 1
                else:
                    confusion_matrix['TN'] += 1
            else:
                if pred_label == '满意' and true_label == '不满意':
                    confusion_matrix['FP'] += 1
                elif pred_label == '不满意' and true_label == '满意':
                    confusion_matrix['FN'] += 1
                else:
                    confusion_matrix['final_match_not_valid'] += 1
                    print(f"Warning: Final match is invalid for item: {item}, true_label: {true_label}, pred_label: {pred_label}")
    
    accuracy = correct / total if total > 0 else 0
    return confusion_matrix, accuracy

def process_jsonl_file(file_path: str) -> None:
    """Process JSONL file and print metrics"""
    results = []
    total_lines = 0
    # Read and parse JSONL file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:100]}...")
                continue
    
    # Calculate metrics
    confusion_matrix, accuracy = calculate_metrics(results)
    
    # Print results
    print("Confusion Matrix:")
    print(f"Total lines: {total_lines}")
    print(f"TP: {confusion_matrix['TP']}")
    print(f"TN: {confusion_matrix['TN']}")
    print(f"FP: {confusion_matrix['FP']}")
    print(f"FN: {confusion_matrix['FN']}")
    print(f"true_label_not_valid: {confusion_matrix['true_label_not_valid']}")
    print(f"pred_label_not_valid: {confusion_matrix['pred_label_not_valid']}")
    print(f"no_pred_response: {confusion_matrix['no_pred_response']}")
    print(f"final_match_not_valid: {confusion_matrix['final_match_not_valid']}")
    print(f"total valid result: {confusion_matrix['TP'] + confusion_matrix['TN'] + confusion_matrix['FP'] + confusion_matrix['FN']}")
    print(f"total not valid result: {confusion_matrix['true_label_not_valid'] + confusion_matrix['pred_label_not_valid'] + confusion_matrix['no_pred_response'] + confusion_matrix['final_match_not_valid']}")
    print(f"\nAccuracy: {accuracy:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python process_jsonl_result.py <jsonl_file_path>")
        sys.exit(1)
    
    process_jsonl_file(sys.argv[1])
