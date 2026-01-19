import os
os.environ["MPLBACKEND"] = "Agg"

# ===== Imports =====
import json
import string
import pandas as pd
from bert_score import BERTScorer



scorer = BERTScorer(
    model_type="aubmindlab/bert-base-arabertv02",
    num_layers=12,
    lang="ar"
)


def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    data = []

    for _, row in df.iterrows():
        data.append({
            "question": str(row.get("question", "")),
            "answer": str(row.get("answer", "")),
            "generatedAnswer":str(row.get("generatedAnswer"))
        })

    return data


def compute_bert_score(truth, prediction):
    if truth == "" or prediction == "":
        return 0.0, 0.0, 0.0

    P, R, F1 = scorer.score([prediction], [truth])

    return (
        float(P.mean().item()),
        float(R.mean().item()),
        float(F1.mean().item())
    )


def main():
    eval_data = load_data_csv("/content/End-to-end-Islamic-Question-Answering-System/data/Evaluation Data/Zeroshot/test_data_zeroshot_10.csv")
    P_scores, R_scores, F1_scores = [], [], []
    for item in eval_data:
        truth = item["answer"]
        prediction =item["generatedAnswer"]
        p, r, f1 = compute_bert_score(truth, prediction)
        P_scores.append(p)
        R_scores.append(r)
        F1_scores.append(f1)
    print(f"Precision: {sum(P_scores)/len(P_scores)}")
    print(f"Recall:    {sum(R_scores)/len(R_scores)}")
    print(f"F1 Score:  {sum(F1_scores)/len(F1_scores)}")
    

if __name__ == "__main__":
    main()

