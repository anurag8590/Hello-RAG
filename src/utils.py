import gc
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from src.embeddings import get_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_pdf_text(pdf_docs):
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.info("Extracted the text successfully.")
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
    
    gc.collect()

    return text


def get_text_chunks(text, chunk_size, chunk_overlap):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        logger.info("Chunking done successfully.")
        gc.collect()
        return chunks
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
    
        gc.collect()

        return []

def get_vector_store(chunks, target_collection, url, api_key):
    try:
        vector_store = Qdrant.from_texts(
            chunks,
            embedding=get_embeddings(),
            url=url,
            api_key=api_key,
            prefer_grpc=False,
            collection_name=target_collection,
            timeout=75
        )
        logger.info("Vector store created successfully.")
        logger.debug(f"Vector store: {vector_store}")
        gc.collect()
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        return None


def get_conversational_chain(vector_store, google_api_key):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vector_store.as_retriever(), memory=memory)
        gc.collect()
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}")
        return None