from langchain_chroma import Chroma
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3.6-flash" , temperature = 0)
persist_directory = "db/chroma_db"
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

class QueryVariations(BaseModel):
    queries: List[str]

#query to search 
query = "How does Tesla make money?"
print(f"query used : {query}")

#generate multiple queries

llm_with_tools = llm.with_structured_output(QueryVariations)

prompt = f"""Generate 3 different variations of this query that would help retrieve relevant documents:

Original query: {query}

Return 3 alternative queries that rephrase or approach the same question from different angles."""

response = llm_with_tools.invoke(prompt)
variations = response.queries
print(f"generated query variations : {len(variations)}")

for i , variation in enumerate(variations , 1):
    print(f"{i} the query {variation}")

#search with each query and store result

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,           
        "fetch_k": 10,    
        "lambda_mult": 0.5  
    }
)

results = []

for i , query in enumerate(variations , 1):
    print(f"\n=== RESULTS FOR QUERY {i}: {query} ===")

    docs = retriever.invoke(query)
    results.append(docs)

    print(f"retrieved {len(docs)} documents")

    for j , doc in enumerate(docs , 1):
        print(f"Document {j} :")
        print(f"{doc.page_content[:150]}..\n")

    print("-" * 50)

print("\n" + "="  * 60)
print("Multi query retrieval completed")

