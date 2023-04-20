import jieba
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import numpy as np
rouge = Rouge()


def compute_metrics(decoded_preds, decoded_labels):
    score_dict = {
        "rouge-1": [],
        "rouge-2": [],
        "rouge-l": [],
        "bleu-4": []
    }
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            if pred:
                hypothesis = list(jieba.cut(str(pred)))
                if len(hypothesis) == 0:
                    hypothesis = ['*****']
            else:
                hypothesis = ['*****']
            if label:
                reference = list(jieba.cut(str(label)))
                if len(reference) == 0:
                    reference = ['*****']
            else:
                reference = ['*****']

            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
        except Exception as e:
            print(e)
            print(pred)
            print(label)

    for k, v in score_dict.items():
        score_dict[k] = float(np.mean(v))
    return score_dict

