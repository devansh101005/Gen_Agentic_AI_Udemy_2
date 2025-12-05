from dotenv import load_dotenv

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

pdf_path= Path(__file__).parent / "Nodejs.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs=loader.load()

#print(docs[10])

#Splitthe docs in smaller chunks

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20,
)
chunks = text_splitter.split_documents(documents=docs)
# print(chunks[0])
# print(chunks[1])

# embedding_model=OpenAIEmbeddings(
#     model="text-embedding-3-large"
# )
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store=QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    prefer_grpc=False,
    collection_name="learning_rag"
)
print("Chunks:", len(chunks))
print("Total points in DB:", vector_store.client.count(collection_name="learning_rag"))


print("Indexing of documents done...")
vec = embedding_model.embed_query("hello world")
print(len(vec))


# #This all above code can be used if we are using openai embedding but its api is paid so below we are doing same thing but with gemini api and embeddings




# from dotenv import load_dotenv
# import os

# from pathlib import Path
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_qdrant import QdrantVectorStore

# load_dotenv()

# pdf_path = Path(__file__).parent / "Nodejs.pdf"
# loader = PyPDFLoader(str(pdf_path))
# docs = loader.load()

# # Split docs
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=20,
# )
# chunks = text_splitter.split_documents(docs)

# embedding_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

# # Qdrant vector store
# vector_store = QdrantVectorStore.from_documents(
#     documents=chunks,
#     embedding=embedding_model,
#     url="http://localhost:6333",
#     collection_name="learning_rag"
# )

# print("Indexing done with Gemini embeddings!")

# yar gemini ka paid hai embedding par open ka free hai ts pmo ho gya 