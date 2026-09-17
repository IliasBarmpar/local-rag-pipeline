# Local RAG Pipeline 🚀

A containerized, privacy-preserving Retrieval-Augmented Generation (RAG) service built with FastAPI, local embeddings, and Ollama.

## 🛠️ Tech Stack
* **Framework:** FastAPI & Uvicorn
* **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
* **LLM Engine:** Ollama (`llama3.2:1b`)
* **Vector Ops:** NumPy (Cosine Similarity)
* **Deployment:** Docker & Docker Compose

## 🧠 Architecture & Guardrails
1. **Embedding & Search:** Queries are embedded into vector space and matched against pre-indexed document chunks using cosine similarity (dot product on normalized embeddings).
2. **Relevance Thresholding:** A similarity guardrail (`threshold >= 0.3`) short-circuits irrelevant queries with a fallback message to prevent hallucinations and save LLM compute.
3. **Grounded Generation:** Context-constrained prompting sent to local Ollama.

## 🚦 Getting Started

### Prerequisites
* Docker & Docker Compose installed
* [Ollama](https://ollama.ai/) running locally with the target model:
  ```bash
  ollama run llama3.2:1b
  ```

### Running with Docker Compose
```bash
docker compose up --build
```
The API will be available at `http://localhost:8000`.

### Running Locally (Without Docker)
```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

---

## 🔌 API Usage

### Interactive Docs
Visit [`http://localhost:8000/docs`](http://localhost:8000/docs) for the interactive Swagger UI.

### Query Endpoint (`POST /ask`)

**Request:**
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I reset my password?"}'
```

**Response (Relevant Query):**
```json
{
  "answer": "To change or recover your account credentials, visit account settings and click reset.",
  "context": "Steps to change or recover your account login credentials: Visit account settings and click reset."
}
```

**Response (Irrelevant Query / Below Threshold):**
```json
{
  "answer": "I don't have enough relevant information to answer that question.",
  "context": ""
}
```