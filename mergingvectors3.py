
import faiss
import pickle
import os
import numpy as np
import logging
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.storage import InMemoryStore

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"
logging.basicConfig(level=logging.INFO)

import pickle

def load_index_metadata(pkl_path):
    """Load metadata from pickle files safely and ensure correct structure."""
    try:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

            processed_data = []
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, tuple) and len(item) == 2:
                        processed_data.append(item)  # Correct format
                    elif isinstance(item, dict):  
                        processed_data.append((item, item.get("page_content", "No content")))  # Convert dict
                    elif isinstance(item, str):  
                        processed_data.append(({"source": "unknown"}, item))  # Treat as content only
                    else:
                        print(f"⚠️ Unexpected metadata format in {pkl_path}: {item}")
            else:
                print(f"⚠️ Unexpected file content in {pkl_path}, expected a list but got {type(data)}")
                return []

            return processed_data
    
    except Exception as e:
        print(f"⚠️ Error loading {pkl_path}: {str(e)}")
        return []


def main():
    # 🔧 FAISS Index Paths
    index_config = {
        "source_1": ("ragsitexmlandindex/faiss_index_m/index.faiss", "ragsitexmlandindex/faiss_index_m/index.pkl"),
        "source_2": ("ragsitexmlandindex/faiss_index_m1/index.faiss", "ragsitexmlandindex/faiss_index_m1/index.pkl"),
        "source_3_transcripts": ("youtubevectors/faiss_transcripts.index", "youtubevectors/transcripts.pkl")
    }

    all_docs = []
    merged_index = None
    index_to_docstore_id = {}

    print("\n🔧 Loading and Merging Indexes:")
    
    for source_name, (idx_path, pkl_path) in tqdm(index_config.items()):
        if not os.path.exists(idx_path):
            print(f"⚠️ Warning: {idx_path} not found. Skipping this index.")
            continue  

        # Load FAISS index
        index = faiss.read_index(idx_path)
        
        # Load metadata
        metadata = load_index_metadata(pkl_path)
        print(f"\n📚 {source_name} contains: {index.ntotal} vectors | {len(metadata)} metadata items")

        if len(metadata) == 0:
            print(f"⚠️ No metadata found for {source_name}. This may cause retrieval issues!")

        # Ensure metadata length matches FAISS index size
        if len(metadata) != index.ntotal:
            print(f"⚠️ Metadata mismatch in {source_name} ({len(metadata)} vs {index.ntotal}). Adjusting metadata...")
            metadata += [{"source": source_name, "info": "Generated metadata"}] * (index.ntotal - len(metadata))

        # Merge FAISS indices and update docstore ID mappings
        if merged_index is None:
            merged_index = index
            all_docs = metadata
            index_to_docstore_id = {i: str(i) for i in range(len(metadata))}
        else:
            vectors = np.array([index.reconstruct(i) for i in range(index.ntotal)])  # Extract vectors
            merged_index.add(vectors)  # Add to merged FAISS
            all_docs += metadata  # Add corresponding metadata
            
            for i in range(index.ntotal):
                doc_id = str(len(index_to_docstore_id))
                index_to_docstore_id[len(index_to_docstore_id)] = doc_id

    # ✅ Final validation
    print(f"\n✅ Final Index: {merged_index.ntotal} vectors")
    print(f"✅ Final Metadata: {len(all_docs)} items")
    print(f"✅ index_to_docstore_id has {len(index_to_docstore_id)} entries")

    # 🔥 Load FAISS vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    class CustomDocstore(InMemoryStore):
       def search(self, doc_id: str):
        """Custom search method to retrieve documents by ID."""
        doc = self.mget([doc_id])  # Get document by ID
        return doc[0] if doc else None  # Return the first document if found


    docstore = CustomDocstore()

    corrected_docs = []
    for i, (meta, content) in enumerate(all_docs):
        if not isinstance(meta, dict):  
            meta = {"source": "generated", "info": str(meta)}  
        corrected_docs.append((str(i), Document(page_content=content, metadata=meta)))

    print(f"✅ Docstore will store {len(corrected_docs)} documents")  
    docstore.mset(corrected_docs)

    vectorstore = FAISS(
        embedding_function=embeddings,
        index=merged_index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})  
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    print("\n🔍 Running batch queries for testing...")

    queries = [
        "How do I apply for a tender on evergabe.de?",
        "Are there any open tenders in Stuttgart?",
        "Where can I find construction tenders?",
        "What are the steps to submit a bid?",
        "Are there IT tenders in Munich?",
        "What are the requirements for submitting a tender in Frankfurt?",
        "What are the requirements for submitting a tender in Munich?"
    ]

    for query in queries:
        response = qa_chain.invoke({"query": query})
        print(f"\n📌 **Question:** {query}")
        print(f"📝 **Answer:** {response['result']}")

if __name__ == "__main__":
    main()
