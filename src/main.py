from in_context_learning.QA import QA
from fastapi import FastAPI
app = FastAPI()
from dotenv import load_dotenv
import os
# Load API KEY
load_dotenv()  
api_key = os.getenv("API_KEY")

#question="لماذا حرّم الله شرب الخمر في القرآن؟"

@app.get('/get_answer/')
async def get_answer(question):
   response={
      "answer":QA(question,api_key)
   }
   return response







