from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


load_dotenv()

persistent_directory = "db/chroma_db"

embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0
)

# Store our conversation as messages
chat_history = []


def ask_question(user_question):
    print(f"\n---Your Question: {user_question} ---")

    #step1 : make the question combined with history
    if chat_history:
            messages = [
                SystemMessage(
                    content="""
                    Given the conversation history, rewrite the new question
                    so that it is completely standalone and searchable.

                    Do not answer the question.
                    Only return the rewritten question.
                    """
                )
            ] + chat_history + [
                HumanMessage(
                    content=f"New question: {user_question}"
                )
            ]
            result = model.invoke(messages)
            search_question = result.content.strip()
    else:
        search_question = user_question

    #step2 : find relevant chunks
    retriever = db.as_retriever(search_kwargs = {"k" : 5})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i , doc in enumerate(docs , 1):
        #show first 2 lines
        lines = doc.page_content.split('\n')[:2]
        preview = '\n'.join(lines)
        print(f"  Doc {i}: {preview}...")

    #step 3 : create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}
    
        Documents:
        {"\n".join([f"- {doc.page_content}" for doc in docs])}
    
        Please provide a clear, helpful answer using only the information from these documents. If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents."
        """

    #step 4 : get the answer
    messages = [
        SystemMessage(content = "You are a helpful assistant that answers questions based on provided documents and conversation history.")
    ] + chat_history + [HumanMessage(content = combined_input)]

    result = model.invoke(messages)
    answer = result.content


    #step 5 : remember this conversation
    chat_history.append(HumanMessage(content = user_question))
    chat_history.append(AIMessage(content = answer))
    print(f"Answer: {answer}")

    return answer

def main():
    print("Welcome to question answer system")
    print("Type 'exit' to quit.")

    while True:
        user_question = input("\n Ask a question")

        if user_question.lower() == 'exit':
            print("Goodbye!")
            break

        answer = ask_question(user_question)

if __name__ == "__main__":
    main()