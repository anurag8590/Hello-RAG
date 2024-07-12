from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from src.embeddings import get_embeddings


def get_pdf_text(pdf_docs):
    
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    print(f"text is {text}")
    print("Extracted the text.......")
    return  text

def get_text_chunks(text,chunk_size,chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    
    print("Chunking Done.......")
    return chunks

def get_vector_store(chunks,target_collection,url,api_key):

    vector_store = Qdrant.from_texts(
        chunks,
        embedding = get_embeddings(),
        url=url,
        api_key=api_key,
        prefer_grpc=False,
        collection_name=target_collection,
        timeout=75
    )

    print("Vector store successfully created..........")
    print(f"vector store = {vector_store}")

    return vector_store

def get_conversational_chain(vector_store,google_api_key):

    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro",google_api_key = google_api_key)
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
    return conversation_chain