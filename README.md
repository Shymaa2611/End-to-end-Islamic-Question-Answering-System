# End-to-end Islamic Question Answering Generation System 
proposed framework is an end-to-end Arabic QA system to answer Islamic questions from Quran and hadith and we incorporate in-context learning with carefully selected demonstrations, guiding the generation process to improve answer quality .

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

git clone https://github.com/Shymaa2611/Islamic-Question-Answering-System.git
cd Islamic-Question-Answering-System

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
 - To Run Demonstrations Retrieval 
 ```bash

!python src/in_Context_learning/run.py

```

 - To Run Islamic Question Answering System With In-context Learning
 ```bash

!python src/in_context_learning/QA.py

```




