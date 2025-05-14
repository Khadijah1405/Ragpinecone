# 🧠 RAGpinecone: FastAPI-Powered RAG API with Pinecone & GPT

This project implements a **Retrieval-Augmented Generation (RAG)** system using **FastAPI**, **OpenAI (GPT-4)**, and **Pinecone** as the vector database. It allows semantic question answering over content vectorized from a company's website or documents.

✅ Uses `FastAPI` to serve a REST API  
✅ Retrieves relevant content from `Pinecone`  
✅ Generates intelligent answers using `GPT-4`  
✅ Deployable on `Render.com` via `render.yaml`

---

## 📁 Project Structure

```
.
├── pineconefastapi.py      # Main FastAPI application
├── render.yaml             # Render deployment file
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-org/Ragpinecone.git
cd Ragpinecone
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the root directory

```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_index_name
PINECONE_REGION=us-east-1
```

---

## ▶️ Running the API Locally

```bash
uvicorn pineconefastapi:app --reload
```

Then open your browser at:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🔁 API Usage

### POST `/query`

**Request:**

```json
{
  "query": "What certifications does the company support?"
}
```

**Response:**

```json
{
  "question": "What certifications does the company support?",
  "answer": "The company supports certifications such as Blue Angel, EU Ecolabel, and EMAS..."
}
```

---

## 🧠 How It Works

1. Content is embedded using OpenAI's `text-embedding-ada-002`.
2. Embeddings are stored in a Pinecone vector index.
3. When queried, the app retrieves relevant documents using similarity search.
4. GPT-4 generates an answer from the retrieved context.

---

## 🛠️ Deployment on Render.com

1. Connect your GitHub repo to Render.
2. Use the `render.yaml` file provided for service configuration.
3. Set the environment variables from your `.env` in Render's dashboard.
4. Deploy and access `/docs` for interactive API testing.

---

## ✅ Use Cases

- Semantic search over web or doc content  
- Internal knowledge retrieval  
- FAQ bots and customer support  
- AI assistant for structured company data
