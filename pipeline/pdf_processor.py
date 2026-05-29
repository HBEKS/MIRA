from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(pdf_path: str):
    """Membaca file PDF dan mengonversinya menjadi FAISS Vector Store secara GRATIS di lokal."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Bangun Vector Store FAISS lokal
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})