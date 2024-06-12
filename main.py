from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from nlp.contexts import contexts

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = "./uploads"

Path(UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# Generator 
generator = pipeline("text-generation", model="gpt2-large")

class UserReply(BaseModel):
    question: str
    user: str

def get_answer(contexts, question, model_name='bert-large-uncased-whole-word-masking-finetuned-squad'):
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    concatenated_contexts = ' '.join(contexts)

    QA_input = {
        'question': question,
        'context': concatenated_contexts
    }

    return nlp(QA_input)['answer']

@app.post("/chatbot-reply")
async def chatbot_reply(request: UserReply):
    try:
        # Load Reply
        answer = get_answer(contexts, request.question)
        
        
        return {"prediction": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

class BlogPost(BaseModel):
    name: str
    doctor: str
    spe: str
    session: str
    notes: str


@app.post("/post-generate")
async def chatbot_reply(request: BlogPost):
    try:
        # Load Reply
        answer = generate_medical_session_post(request.name, request.doctor, request.spe, request.session, request.notes)
        
        
        return {"prediction": answer}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



def generate_medical_session_post(patient_name, doctor_name, doctor_specialization, session_topic, session_notes):
    question = f"Today, {patient_name} had a medical session with {doctor_name}, a {doctor_specialization}, regarding {session_topic}. Here are the session notes: {session_notes}"

    response = generator(
        question, max_length=200, num_return_sequences=1, temperature=0.8
    )
    post = response[0]["generated_text"]
    return post

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port="8000")
