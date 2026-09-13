import time
from docx import settings
from pinecone import Pinecone,ServerlessSpec

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_Settings

settings=get_Settings()

_embeddings=None
_vectorstore=None


EMBEDDING_DIMENSIONS={
    "text-embedding-3-small":1536,
    "text-embedding-3-large":3072,
    "text-embedding-ada-002":1536,
    "all-minilm-l6-v2":384,
}

#Embedding model configuration
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        if not settings.openai_api_key:
            raise RuntimeError("OpenAPI key is missing")
        
        _embeddings=OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        
    return _embeddings



#pinecone configuration
def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("Pinecone_API_KEY is missing")
    
    desired_dimension=1536
    pc=Pinecone(api_key=settings.pinecone_api_key)
    names=[x["name"] for x in pc.list_indexes()]
    
    if settings.pinecone_index_name in names:
        #It will give the index info (metadata)
        index_info=pc.describe_index(settings.pinecone_index_name)
        current_dimension=getattr(index_info,"dimension",None)
        if current_dimension is None and isinstance(index_info,dict):
            current_dimension=index_info.get("dimension")
            
        #if dimensions are not mathced it will delete the index
        if current_dimension is not None and current_dimension!=desired_dimension:
            pc.delete_index(name=settings.pinecone_index_name)
            while settings.pinecone_index_name in [x["name"] for x in pc.list_indexes()]:
                time.sleep(1)
                
                
    if settings.pinecone_index_name not in [x["name"] for x in pc.list_indexes()]:
        #It will create the new index
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws",region="us-east-1"),
        )
        
        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            time.sleep(1)
            
    #It will return the index object     
    return pc.Index(settings.pinecone_index_name)



#Storing into Pinecone
def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index=ensure_index()
        _vectorstore=PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
        
    return _vectorstore


#Retrieving from pinecone
def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k":settings.top_k})



def add_documents(chunks):
    store=get_vectorstore()
    return store.add_documents(chunks)
