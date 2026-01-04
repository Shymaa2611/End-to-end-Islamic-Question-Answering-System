import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
# ===== Load AraBERT Modern =====
MODEL_NAME = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

# ===== Mean Pooling =====
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
        input_mask_expanded.sum(dim=1), min=1e-9
    )

# ===== Encode text =====
def encode(text):
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        output = model(**encoded)
    embedding = mean_pooling(output, encoded["attention_mask"])
    return embedding.numpy()

# ===== Cosine Similarity =====
def cosine_sim(text1, text2):
    e1 = encode(text1)
    e2 = encode(text2)
    return cosine_similarity(e1, e2)[0][0]


def load_data_csv(file_path):
    df = pd.read_csv(file_path)
    data = []

    for _, row in df.iterrows():
        data.append({
            "answer": str(row.get("answer", "")), 
            "generatedAnswer": str(row.get("generatedAnswer", "")),   
        })

    return data
def main():
    scors=[]
    eval_data = load_data_csv("/content/test_data_oneshot_70.csv")
    for item in eval_data:
        answer = item["answer"]
        generated=item["generatedAnswer"]
        score = cosine_sim(answer, generated)
        scors.append(score)

    print("average similarity = ",sum(scors)/len(scors))
        

     

if __name__ == "__main__":
    main()

