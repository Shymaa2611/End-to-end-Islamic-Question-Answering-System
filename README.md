# End-to-End Islamic Question Answering System

This repository presents an end-to-end Arabic Islamic Question Answering (QA) system designed to answer religious questions using authoritative Islamic sources, namely the Holy Qur’an and Hadith.
The system integrates retrieval-augmented generation with in-context learning (ICL), where carefully selected demonstrations guide the answer generation process to improve accuracy, relevance, and faithfulness to source texts.

![Proposed End-to-End Islamic Question Answering System](images/framework.png)

## Methodology

### Retrieval

Given an input question, the system retrieves the top-K relevant passages from a curated collection of Qur’anic verses and Hadith texts.
Both the question and candidate passages are encoded using a sentence embedding model, and cosine similarity is computed to rank passages based on semantic relevance.

### Reranking
to re-rank the retrieved passages, we employed NAMAA Space GATE Reranker V1 (GATE) the baseline model. GATE is a cross-encoder–based model built on AraBERT and trained using the Arabic Triplet Matryoshka strategy , which enabled it to learn fine-grained relevance distinctions between candidate passages while remaining computationally efficient. By jointly encoding question–passage pairs, the reranker leveraged cross-attention mechanisms to capture contextual and semantic interactions, producing more accurate relevance scores than retrieval-only methods .

###  In-Context Learning
In-Context Learning (ICL) enables a language model to perform a task by conditioning on a small set of input–output demonstrations included directly in the prompt, without updating model parameters.
While effective, ICL performance is highly sensitive to demonstration selection.
To address this, we adopt a retrieval–reranking-based demonstration selection strategy, inspired by our passage retrieval pipeline:
-For a given input question, candidate passages are retrieved from the training set using a dense 
 retriever.
- These candidates are reranked using the GATE cross-encoder to compute fine-grained relevance scores.
- The top-ranked passages are selected, and their associated question–answer pairs are extracted from the  
  training data.
- These pairs are used as in-context demonstrations.
This approach ensures that demonstrations are not only semantically similar to the input question, but also highly relevant at the interaction level, resulting in more effective and informative ICL.

###  Answer EXtraction

Final answer extract is performed using a large language model (LLM) from Mistral AI.
The model is explicitly constrained to extract answers only from the retrieved and reranked evidence, ensuring grounded and reliable outputs.
Answer generation is formulated as an answer extraction task using a structured prompt with two components:

- In-Context Demonstrations ,Each demonstration follows a fixed structure:
Context: Retrieved passages
Question: Demonstration query
Answer: Ground-truth response

- Test Instance
Question: User input
Context: Top-K reranked passages
Answer: Left empty for model generation
By enforcing a consistent context–question–answer format, the model is guided to produce answers that are faithful to the retrieved evidence, significantly reducing hallucinations and improving answer quality.

![Prompt Template](images/template.png)


## Installation & Requirements

### Requirements
- Python 3.10
- MistralAI API
- Additional libraries:

``` bash
!pip install huggingface_hub==0.13.4
!pip install -U sentence-transformers
!pip install faiss-cpu
!pip install -U google-generativeai
!pip install -U google-genai

```

### Installation
1- Clone the repository:

```bash

git clone https://github.com/Shymaa2611/End-to-end-Islamic-Question-Answering-System-Generation.git
cd End-to-end-Islamic-Question-Answering-System-Generation

```

2- Create and activate a virtual environment (optional but recommended):
```bash

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```
3- Install dependencies:
```bash

pip install -r requirements.txt

```

4- Run System:
 - To Run Demonstrations Selection
 ```bash

!python src/in_Context_learning/run.py

```

 - To Run Islamic Question Answering System With In-context Learning
 ```bash

!python src/in_context_learning/QA.py

```

5- Evaluate System

```bash

!python src/Evaluation/eval.py

```



