from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()



persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory = persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

# Query to test
query = "How much did Microsoft pay to acquire GitHub?"
# query = "How do you plant tomatoes in a garden?"
print(f"Query: {query}\n")

#Method 1

print("=========================================")
print("METHOD 1: Basic Retrieval")
print("=========================================")

retriever = db.as_retriever(search_kwargs = {"k" : 3})

docs = retriever.invoke(query)
print(f"retrieved {len(docs)} documents")

for i, doc in enumerate(docs[:2]):
    print(f"Document{i+1}")
    print(doc.page_content)
    


#method2
print("=========================================")
print("METHOD 2: Similarity with base retrival ")
print("=========================================")

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.3  
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (threshold: 0.3):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")


#method 3
print("=========================================")
print("METHOD 3:  Maximum Marginal Relevance (MMR) ")
print("=========================================")

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,           
        "fetch_k": 10,    
        "lambda_mult": 0.5  
    }
)

docs = retriever.invoke(query)
print(f"Retrieved {len(docs)} documents (λ=0.5):\n")

for i, doc in enumerate(docs, 1):
    print(f"Document {i}:")
    print(f"{doc.page_content}\n")