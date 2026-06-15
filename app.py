from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask import Flask,request, jsonify
import requests
import os
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")
app = Flask(__name__)
model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    api_key=groq_api_key,

    # other params...
)

from langchain_docling.loader import DoclingLoader

FILE_PATH = 'data/Dental_Clinic_RAG_Documents.docx'

loader = DoclingLoader(file_path=FILE_PATH)
docs = loader.load()

from langchain_community.vectorstores.utils import filter_complex_metadata

docs = filter_complex_metadata(docs)
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

query_embedding = embeddings.embed_query("what is dental hygiene")
doc_embeddings = embeddings.embed_documents(
    [
      doc.page_content for doc in docs
    ]
)
from langchain_chroma import Chroma

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)
def ask_rag(question):
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"""
    You are a helpful dental clinic assistant.
    Answer the user's question using ONLY the context below.

    Context:
    {context}

    Question:
    {question}
    """
    response = model.invoke(prompt)
    return response.content

def send_whatsapp_message(to, message):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message}
    }
    requests.post(url, headers=headers, json=payload)

# ── Webhook verification (Meta calls this once to verify your URL) ─
@app.route("/webhook", methods=["GET"])
def verify():
    if request.args.get("hub.verify_token") == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return "Invalid token", 403

# ── Webhook receiver (Meta calls this every time user sends message)
@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.json
    try:
        message = data["entry"][0]["changes"][0]["value"]["messages"][0]
        user_number = message["from"]      # who sent the message
        user_text = message["text"]["body"] # what they said
        reply = ask_rag(user_text)          # get answer from RAG
        send_whatsapp_message(user_number, reply)  # send reply back
    except (KeyError, IndexError):
        pass  # ignore non-message events (delivery receipts, etc.)
    return jsonify({"status": "ok"})

# ── Run the Flask server ───────────────────────────────────────────
if __name__ == "__main__":
    app.run(port=5000)
