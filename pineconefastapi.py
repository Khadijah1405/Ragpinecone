import os
import logging
import traceback
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# --- Load environment variables ---
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_REGION = os.getenv("PINECONE_REGION")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
logging.basicConfig(level=logging.INFO)

# --- Pinecone Client Init ---
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# --- FastAPI App ---
app = FastAPI(title="RAG with Pinecone")
qa_chain = None

class QueryRequest(BaseModel):
    query: str

def build_qa_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, text_key="text")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

@app.on_event("startup")
def startup_event():
    global qa_chain
    logging.info("🔧 Initializing Pinecone-based RAG pipeline...")
    qa_chain = build_qa_chain()
    logging.info("✅ Ready.")

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        response = qa_chain.invoke({"query": request.query})
        return {"question": request.query, "answer": response["result"]}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
