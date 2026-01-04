# End-to-End Islamic Question Answering Generation System

This repository presents an end-to-end Arabic Islamic Question Answering (QA) system designed to answer religious questions using authoritative Islamic sources, namely the Holy Qur’an and Hadith.
The system integrates retrieval-augmented generation with in-context learning (ICL), where carefully selected demonstrations guide the answer generation process to improve accuracy, relevance, and faithfulness to source texts.
![Proposed End-to-end Islamic Question Answering Generation System](images\framework.png)
## Methodology

### Retrieval

Given an input question, the system retrieves the top-K relevant passages from a curated collection of Qur’anic verses and Hadith texts.
Both the question and candidate passages are encoded using a sentence embedding model, and cosine similarity is computed to rank passages based on semantic relevance.

### Reranking
To refine the retrieved results, we employ NAMAA Space GATE Reranker V1 (GATE), a cross-encoder model fine-tuned by NUR for the IslamicEval 2025 Shared Task.
GATE is built upon AraBERT and utilizes the Arabic Triplet Matryoshka training strategy, achieving strong performance while remaining computationally efficient.

For zero-answer detection, a threshold-based decision mechanism is applied. If all reranked passages score below a predefined threshold, the question is classified as having no valid answer, and the system returns −1.
###  In-Context Learning
In-Context Learning (ICL) is a learning paradigm in which a language model performs a task by conditioning on a small set of input–output examples (demonstrations) provided within the prompt, without requiring any parameter updates. This enables a single language model to handle diverse NLP tasks through inference alone, making ICL an attractive alternative to supervised fine-tuning . However, previous studies have shown that the performance of ICL is highly sensitive to the choice of demonstrations . As a result,recent work has focused on developing effective demonstration retrieval strategies for in-context learning. Existing approaches to demonstration retrieval generally fall into two categories. The first relies on off-the-shelf retrieval models, such as BM25 or Sentence-BERT, to retrieve demonstrations that are lexically or semantically similar to the test query. While these methods provide empirical    improvements, they rely primarily on coarse similarity measures. In this work, we adopt a retrieval–reranking-based demonstration selection strategy, inspired by our passage retrieval pipeline Given an input question, we first retrieve a set of candidate passages from the training data using a dense retriever. To further refine the selection, we apply a GATE cross-encoder reranker to re-evaluate the relevance between the input question and each candidate passage. The reranker assigns fine-grained relevance scores, and the top-ranked passages are selected as most relevant candiated passages . For
each relevant candiated passages , we extract the associated question–answer pair from the training
set and treat it as a candidate in-context demonstration.These demonstrations are then incorpo-
rated into the prompt and provided to the language model for inference.This retrieval–reranking
approach enables the selection of demonstrations that are not only semantically similar to the in-
put question, but also highly relevant at the interaction level, resulting in more informative and
effective in-context learning.

###  Answer Generation

Final answer generation is performed using a large language model (LLM) from Mistral AI.
The model is explicitly constrained to generate answers only from the retrieved and reranked evidence, ensuring grounded and reliable outputs.
Answer generation is formulated as an answer extraction task using a structured prompt with two components:
In-Context Demonstrations ,Each demonstration follows a fixed structure:

Context: Retrieved passages
Question: Demonstration query
Answer: Ground-truth response

Test Instance

Question: User input
Context: Top-K reranked passages
Answer: Left empty for model generation
By enforcing a consistent context–question–answer format, the model is guided to produce answers that are faithful to the retrieved evidence, significantly reducing hallucinations and improving answer quality.

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

5- Run System:
 - To Run Demonstrations Selection
 ```bash

!python src/in_Context_learning/run.py

```

 - To Run Islamic Question Answering System With In-context Learning
 ```bash

!python src/in_context_learning/QA.py

```




