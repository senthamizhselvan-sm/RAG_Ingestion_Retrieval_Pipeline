import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# Load documents
def load_documents(docs_path="docs"):
    """Load documents from the specified directory."""
    print(f"loading documents from {docs_path}")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(
            f"The specified path '{docs_path}' does not exist."
        )

    loader = DirectoryLoader(
        docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents = loader.load()

    if len(documents) == 0:
        raise ValueError(
            f"No documents found in the specified path '{docs_path}'."
        )

    for i, doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"content_length: {len(doc.page_content)}")
        print(f"content: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks."""
    print("Splitting documents into chunks")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1}:")
            print(f"Source: {chunk.metadata['source']}")
            print(f"content_length: {len(chunk.page_content)}")
            print(f"content: {chunk.page_content[:100]}...")
            print(f"metadata: {chunk.metadata}")

        if len(chunks) > 2:
            print(f"\nTotal chunks created: {len(chunks)}")

    return chunks


def create_vectorstore(chunks, persistant_directory):
    """Create a vector store for document chunks."""
    print("creating embedding and storing in chromadb")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persistant_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(
        f"Persisted vector store with "
        f"{vectorstore._collection.count()} documents"
    )

    return vectorstore


def main():
    """Main Data Ingestion pipeline."""
    print("RAG documents Ingestion pipeline")

    docs_path = "docs"
    persistant_directory = "db/chroma_db"

    # Check if the persistent directory exists
    if os.path.exists(persistant_directory):

        print("vector store already exists, no need reprocess it")

        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=persistant_directory,
            embedding_function=embedding_model,
            collection_metadata={"hnsw:space": "cosine"}
        )

        print(
            f"Loaded existing vector store with "
            f"{vectorstore._collection.count()} documents"
        )

        return vectorstore

    print(
        "Persistent directory does not exist, "
        "creating it and processing documents"
    )

    # Step 1: Load Documents
    documents = load_documents(docs_path)

    # Step 2: Split documents into chunks
    chunks = split_documents(documents)

    # Step 3: Create embeddings and store in vector store
    vectorstore = create_vectorstore(
        chunks,
        persistant_directory
    )

    return vectorstore


if __name__ == "__main__":
    main()