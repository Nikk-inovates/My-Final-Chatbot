from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from src.load_pdf import load_pdf_text
from src.embed_text import split_text, embed_chunks, save_faiss_index, load_faiss_index
from src.chatbot import setup_deepseek, ask_question, search_chunks
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import shutil

# 🌱 Load environment variables
load_dotenv()

app = FastAPI()
PDF_DIR = "data"
os.makedirs(PDF_DIR, exist_ok=True)

# 🔗 Load embedding model only once
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 🤖 Chat model (static for now, could be dynamic later)
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Uploads and processes a PDF file.
    """
    try:
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 📄 Load and chunk
        text = load_pdf_text(file_path)
        chunks = split_text(text)
        _, index, _, chunk_list = embed_chunks(chunks)

        # 💾 Save FAISS index
        save_faiss_index(index, chunk_list)

        return {"message": "✅ PDF uploaded and processed successfully!"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to process PDF: {e}"})


@app.post("/ask-question/")
async def ask_question_api(question: str = Form(...)):
    """
    Answers a question based on the indexed PDF.
    """
    try:
        index, chunk_list = load_faiss_index()
        top_chunks = search_chunks(embedding_model, index, chunk_list, question)
        response = ask_question(MODEL_NAME, top_chunks, question)
        return {"answer": response}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"❌ Failed to get answer: {e}"})


@app.get("/")
def root():
    return {"message": "🚀 PDF Chatbot API is running!"}
