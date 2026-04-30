# !pip install faiss-gpu-cu11==1.10.0
# !pip install --upgrade sentence_transformers

import pandas as pd
import json
import faiss
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer
import ast
import re
from huggingface_hub import snapshot_download
from retrieve_demonstrations import template
import requests
import json

from dotenv import load_dotenv
import os

snapshot_download(
    repo_id="SeragAmin/NAMAA-retriever-cosine-final_60-90",
    repo_type="model",
    local_dir="retriever_model",
    allow_patterns="NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985/*"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# LOAD RETRIEVAL MODEL
retrieval_model = SentenceTransformer("retriever_model/NAMAA-retriever-cosine-final_contrastive_ara_top70/checkpoint-1985")
retrieval_tokenizer = retrieval_model.tokenizer
retrieval_model.to(device)
retrieval_model.eval()
model = CrossEncoder("yoriis/GTE-tydi-quqa-haqa")
diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670]')

# Encoding Function
def get_embedding(text):
    with torch.no_grad():
        emb = retrieval_model.encode(text, convert_to_numpy=True, device=device)
    return emb

# Indexing Function

def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# Load Quran data files
quran_passages = []
with open("/content/End-to-end-Islamic-Question-Answering-System/data/QH-QA-25_Subtask2_QPC_v1.1.tsv", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            passage_id = parts[0]
            passage_text = parts[1]
            quran_passages.append({"text": passage_text, "source": "quran", "id": passage_id})

# Load Hadith data files
hadith_passages = []
with open("/content/End-to-end-Islamic-Question-Answering-System/data/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = ast.literal_eval(line.strip())
            cleaned_text = diacritics_pattern.sub('', item['hadith'])
            hadith_passages.append({
                  "text": cleaned_text,
                  "source": "hadith",
                  "id": item['hadith_id']
            })
        except Exception as e:
            print(f"Skipping invalid line: {e}")

all_passages = quran_passages + hadith_passages
print(f" Loaded total passages: {len(all_passages)}")

quran_texts = [p["text"] for p in quran_passages]
hadith_texts = [p["text"] for p in hadith_passages]


# Quran & Hadith Embeddings
quran_embeddings = retrieval_model.encode(
    quran_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
hadith_embeddings = retrieval_model.encode(
    hadith_texts,
    convert_to_numpy=True,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Quran & Hadith Indexing 
quran_index = build_faiss_index(quran_embeddings)
hadith_index = build_faiss_index(hadith_embeddings)

# Return list of  Relevant Quran & Hadith Passages with Score
def search(query, k_quran=50, k_hadith=20):
    query_emb = get_embedding(query)

    # Search separately
    D_q, I_q = quran_index.search(np.array([query_emb]), k_quran)
    D_h, I_h = hadith_index.search(np.array([query_emb]), k_hadith)

    results = []

    for i, score in zip(I_q[0], D_q[0]):
        passage = quran_passages[i]
        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "quran",
            "text": passage["text"]
        })

    for i, score in zip(I_h[0], D_h[0]):
        passage = hadith_passages[i]
        results.append({
            "score": float(score),
            "id": passage["id"],
            "source": "hadith",
            "text": passage["text"]
        })

    # Optionally, sort by score (before reranking)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    return results

# Return list of Rerank Relevant Quran & Hadith Passages with Score
def predict_Question_rerank_crossencoder(question, model, search_fn, k_retrieve=70, score_threshold=0.15, max_returned=20):
    all_results = []
    # List of Quran & Hadith Passages with Score
    retrieved = search_fn(question)

    # get texts of retrieved passages 
    candidate_texts = [r["text"] for r in retrieved]
 
    # rerank retrieved Quran & Hadith Passages based on the most relevant for question
    reranked = model.rank(query=question, documents=candidate_texts)
   
    # handle the no-answer questions
    filtered = [item for item in reranked if item['score'] >= score_threshold]
    filtered = sorted(filtered, key=lambda x: x['score'], reverse=True)[:max_returned]
   
    # check if zero answer
    if not filtered:
            all_results.append({
               
                "لا توجد اجابة"
            })
        
    # collect top texts
    for item in filtered:
        corpus_id = item['corpus_id']
        if corpus_id < len(candidate_texts):
            all_results.append(candidate_texts[corpus_id])

    return all_results

# Answer Extraction Function
def QA(question,api_key):
    # get the rerank passages relevent for question
    candiated_passages=predict_Question_rerank_crossencoder(question, model, search_fn=search, k_retrieve=70)
    context = "\n".join([f"Passage {i+1}: {p}" for i, p in enumerate(candiated_passages)])

    #create Answer Extraction Template with few examples
    prompt=template(question,context)
    
    # Matral ai API
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "<YOUR_SITE_URL>", # Optional. Site URL for rankings on openrouter.ai.
        "X-Title": "<YOUR_SITE_NAME>", # Optional. Site title for rankings on openrouter.ai.
    },
    data=json.dumps({
        "model":"mistralai/devstral-2512:free",
        "messages": [
        {
            "role": "user",
            "content": prompt
        }
        ]
    })
    )
    response.raise_for_status()  

    return response.json()["choices"][0]["message"]["content"]
 

   








 
