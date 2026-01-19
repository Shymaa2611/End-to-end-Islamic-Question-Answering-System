import string
import pandas as pd
def normalize_text(s):
    """remove punctuation, some stopwords and extra whitespace."""
    def remove_stopWords(text):
        terms = []
        stopWords = {'من', 'الى', 'إلى', 'عن', 'على', 'في', 'حتى'}
        for term in text.split():
            if term not in stopWords:
                terms.append(term)
        return " ".join(terms)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        # Arabic punctuation
        exclude.add('،')
        exclude.add('؛')
        exclude.add('؟')
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_stopWords(remove_punc(s)))


def exact_match_score(prediction, ground_truth,):
    if len(prediction) == 0: 
        return 0
    return (normalize_text(prediction) == normalize_text(ground_truth))

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


def main():
    eval_data = load_data_csv("/content/test_data_oneshot_20.csv")
    EM = []
    for item in eval_data:
        truth = item["answer"]
        prediction =item["generatedAnswer"]
        em=exact_match_score(prediction,truth)
        EM.append(em)
       
    print(f"EM:  {sum(EM)/len(EM)}")
    
if __name__ == "__main__":
    main()