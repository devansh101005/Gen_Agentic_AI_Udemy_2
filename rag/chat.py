from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

load_dotenv()

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag",
    embedding=embedding_model,
)

client=OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Take user input
user_query = input("Ask something: ")

# Relevant chunks from the vector db
search_results = vector_db.similarity_search(query=user_query)

context = "\n\n\n".join([f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}" for result in search_results])


SYSTEM_PROMPT = f"""
 You are a helpfull AI Assistant who answeres user query based on the available context retrieved from a PDF file along with page_contents and page number.

 You should only ans the user based on the following context and navigate the user to open the right page number to know more.

 Context:
 {context}
"""



response=client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role":"system","content": SYSTEM_PROMPT},
        {"role":"user","content":user_query}
    ]
)

print(f"Agent: {response.choices[0].message.content}")

