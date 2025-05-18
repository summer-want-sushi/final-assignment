import os

from fastapi import FastAPI, UploadFile, File

from FraudDetectionAgent import CodeAgent

app = FastAPI()


@app.post("/detect-fraud/")
async def detect_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "temp_transactions.csv"

    with open(temp_path, "wb") as f:
        f.write(contents)

    agent = CodeAgent(data_path=temp_path)
    frauds = agent.run()

    os.remove(temp_path)
    return {"flagged_transactions": frauds}


from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from FraudDetectionAgent import FraudDetectionAgent
import pandas as pd
import os

app = FastAPI()


# Basic chat message schema
class ChatMessage(BaseModel):
    message: str


@app.post("/chat/")
def chat_with_agent(msg: ChatMessage):
    user_input = msg.message.lower()

    if "fraud" in user_input:
        return {"response": "You can upload a CSV file at /detect-fraud/ to check for fraudulent transactions."}
    elif "hello" in user_input or "hi" in user_input:
        return {"response": "Hello! I can help you detect transaction frauds. Ask me how."}
    else:
        return {"response": "I'm still learning! Try asking about fraud detection or uploading a CSV."}


@app.post("/detect-fraud/")
async def detect_fraud(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "temp_transactions.csv"
    with open(temp_path, "wb") as f:
        f.write(contents)

    agent = FraudDetectionAgent(data_path=temp_path)
    frauds = agent.run()
    os.remove(temp_path)
    return {"flagged_transactions": [f.to_dict() for f in frauds]}


from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from FraudDetectionAgent import CodeAgent
import pandas as pd
import os

app = FastAPI()

# Enable CORS for browser frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    message: str


@app.post("/chat/")
def chat_with_agent(msg: ChatMessage):
    text = msg.message.lower()
    if "fraud" in text:
        return {"response": "You can upload a CSV of transactions below and I'll tell you which ones look suspicious."}
    elif "hello" in text:
        return {"response": "Hi there! I'm your fraud detection assistant. Upload a CSV to get started."}
    else:
        return {
            "response": "I'm here to help with transaction fraud detection. Try asking about fraud or upload your data."}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        temp_file = "temp_data.csv"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        agent = FraudDetectionAgent(data_path=temp_file)
        frauds = agent.run()
        os.remove(temp_file)

        if not frauds:
            return {"response": "✅ No fraud detected!"}

        summary = "\n".join([
            f"- ID: {f.get('transaction_id')}, Amount: ${f.get('amount')}"
            for f in frauds[:5]
        ])
        return {"response": f"⚠️ Detected {len(frauds)} suspicious transactions:\n{summary}"}

    except Exception as e:
        return {"response": f"❌ Error processing file: {str(e)}"}