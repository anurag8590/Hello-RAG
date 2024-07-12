import streamlit as st
from src.utils import *
from src.embeddings import get_embeddings

GEMINI_API_KEY = None
QDRANT_API_KEY = None
QDRANT_HOST = None

st.set_page_config("RAG | Qdrant & Gemini")
st.header("RAG with Qdrant 🚀 + Gemini ❇️ ")

with st.sidebar:

    GEMINI_API_KEY = st.text_input("Enter Gemini ❇️ API:")
    QDRANT_API_KEY = st.text_input("Enter Qdrant 🚀 API:")
    QDRANT_HOST = st.text_input("Enter Qdrant 🚀 endpoint:")


genai.configure(api_key=GEMINI_API_KEY)

client = QdrantClient(
    url = QDRANT_HOST,
    api_key=QDRANT_API_KEY
)


def user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("You 🐼 : ", message.content)
        else:
            st.write("Gemini ✨: ", message.content)

def create_database():
    
    client = QdrantClient(
    url = QDRANT_HOST,
    api_key=QDRANT_API_KEY
    )

    with st.sidebar:

        chunk_size = st.slider("Select the chunk size:",min_value=512, max_value=1500)
        chunk_overlap = st.slider("Select the overlap:",min_value=20, max_value=300)

    pdf_docs = st.file_uploader("Select PDF Files & Click on the Submit Button 😉", type=(".pdf"), accept_multiple_files=True)

    target_collection = st.text_input("Enter the Collection name 📒:")

    if st.button("Submit"):

        with st.spinner("Processing⌛..."):

            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text,chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            get_vector_store(text_chunks,target_collection,url=QDRANT_HOST,api_key=QDRANT_API_KEY)
            
            st.success("Done")


def rag():    

    with st.sidebar:
        if 'collections' not in st.session_state:
            st.session_state.collections = []
        if 'selected_collection' not in st.session_state:
            st.session_state.selected_collection = None

        if st.button("Show Collections in VectorDB 🔎"):
            
            client = QdrantClient(
            url = QDRANT_HOST,
            api_key=QDRANT_API_KEY
            )

            collections = client.get_collections().collections
            st.session_state.collections = [collection.name for collection in collections]

            if st.session_state.selected_collection not in st.session_state.collections:
                st.session_state.selected_collection = st.session_state.collections[0] if st.session_state.collections else None

        if st.session_state.collections:
            st.session_state.selected_collection = st.selectbox(
                "Choose the collection: 🤔", 
                st.session_state.collections,
                index=st.session_state.collections.index(st.session_state.selected_collection) if st.session_state.selected_collection else 0
            )
        
        if st.session_state.selected_collection:
            st.write(f"Selected collection: {st.session_state.selected_collection}")
            
            client = QdrantClient(
            url = QDRANT_HOST,
            api_key=QDRANT_API_KEY
            )

            vector_store = Qdrant(
                client=client,
                collection_name=st.session_state.selected_collection,
                embeddings=get_embeddings()  
            )
            
            if "conversation" not in st.session_state or st.session_state.conversation is None:
                st.session_state.conversation = get_conversational_chain(vector_store,google_api_key=GEMINI_API_KEY)

    st.title("Information Retrieval 🕵")
    user_question = st.text_input("Ask me a Question 😄")
    
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = []

    if user_question:
        user_input(user_question)
    
    st.text("Tip💡: Make Sure to clear the input before changing the collection name!!")


option = st.sidebar.selectbox("Choose an action ✅", ("Create a Database", "Information Retrieval"))

if option == "Create a Database":
    create_database()
elif option == "Information Retrieval":
    rag()
