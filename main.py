from contextlib import asynccontextmanager
import numpy as np
import ollama
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 1. Resource storage 🧠
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load embedding model once at startup 🚀
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ml_models["embed_model"] = model

    # Pre-encode documents once 📄
    documents = [
        "Steps to change or recover your account login credentials: Visit account settings and click reset.",
        "Account billing cycles and subscription tiers are processed on the first of each month.",
    ]
    ml_models["documents"] = documents
    ml_models["doc_vecs"] = model.encode(documents, normalize_embeddings=True)

    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# 2. Request / Response schemas 📋
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    context: str

# 3. The RAG endpoint 🔌
@app.post("/ask", response_model=QueryResponse)
async def ask(payload: QueryRequest):
    # Retrieval 🔍
    query_vec = ml_models["embed_model"].encode(payload.query, normalize_embeddings=True)
    scores = np.dot(ml_models["doc_vecs"], query_vec)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    print(f"DEBUG: Best similarity score was {best_score}")
    if best_score < 0.3:
        return QueryResponse(
            answer="I don't have enough relevant information to answer that question.",
            context=""
        )
    
    retrieved_context = ml_models["documents"][best_idx]

    # Augmentation 📝
    rag_prompt = f"""Answer the question using only the context below.

Context:
{retrieved_context}

Question:
{payload.query}

Answer:"""

    # Generation 💬
    response = ollama.generate(model="llama3.2:1b", prompt=rag_prompt)

    return QueryResponse(
        answer=response["response"],
        context=retrieved_context
    )