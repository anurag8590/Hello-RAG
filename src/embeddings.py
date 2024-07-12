from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():

    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return embeddings