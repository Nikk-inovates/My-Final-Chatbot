import os
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import datetime  # For timestamping logs

# Load environment variables from .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("❌ OPENROUTER_API_KEY not found in .env file.")

MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

DEBUG = False  # Set to True to debug prompt


def setup_deepseek():
    """Returns the DeepSeek model name."""
    return MODEL_NAME


def ask_question(model_name, context_chunks, user_question):
    """
    Uses DeepSeek API to generate a response from context + question.
    """
    if not context_chunks:
        return "⚠️ No context available to answer the question."

    context = "\n\n".join(context_chunks)
    prompt = (
        "You are a helpful AI assistant. Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_question}"
    )

    if DEBUG:
        print("Prompt to DeepSeek:\n", prompt)

    try:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that answers based on given context."},
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(
            url=OPENROUTER_API_URL,
            headers=HEADERS,
            json=payload
        )

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"❌ DeepSeek API error {response.status_code}: {response.text}"

    except Exception as e:
        return f"❌ Unexpected error: {e}"


def search_chunks(embedding_model, index, chunks, query, top_k=3):
    """
    Uses FAISS to retrieve top-k chunks relevant to the query.
    """
    if not query.strip():
        return [], []

    try:
        query_embedding = embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = index.search(query_embedding, top_k)
        top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
        top_indices = [i for i in I[0] if i < len(chunks)]

        return top_chunks, top_indices

    except faiss.FaissException as faiss_err:
        print(f"❌ FAISS error during search: {faiss_err}")
        return [], []
    except Exception as e:
        print(f"❌ Unexpected error in search_chunks: {e}")
        return [], []


def log_chat_to_history(question, answer, filename="logs/chat_history.json"):
    """
    Logs each Q&A interaction to a JSON file for persistent history.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        chat_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer
        }

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except json.JSONDecodeError:
                history = []
        else:
            history = []

        history.append(chat_entry)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)

    except Exception as e:
        print(f"⚠️ Failed to log chat history: {e}")


def log_feedback(chunk_ids, question, answer, rating, comment=None, filename="logs/feedback.json"):
    """
    Logs feedback from the user on AI responses.
    """
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        feedback_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "chunk_ids": chunk_ids,
            "question": question,
            "answer": answer,
            "rating": rating,
            "comment": comment
        }

        if os.path.exists(filename):
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    feedback_data = json.load(f)
            except json.JSONDecodeError:
                feedback_data = []
        else:
            feedback_data = []

        feedback_data.append(feedback_entry)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, indent=4)

    except Exception as e:
        print(f"⚠️ Failed to log feedback: {e}")
