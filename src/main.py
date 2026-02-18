from in_context_learning.QA import QA


question="لماذا حرّم الله شرب الخمر في القرآن؟"

def main():
   answer= QA(question)
   return answer

if __name__ == "__main__":
   answer=main()
   print(f"\n question : {question} \n answer : {answer}")
   