from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

persistance_directory = "db/chroma_db"

#Load Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(
    persist_directory=persistance_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

#Search for relevant documents
query = "How much did Microsoft pay to acquire GitHub?"


#set retriver
retriever = db.as_retriever(search_kwargs={"k" : 5});

#invoke relevant docs
relevant_docs = retriever.invoke(query)

print(f"User query : {query}")

for i , doc in enumerate(relevant_docs):
    print(f"Document {i+1} : \n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"
