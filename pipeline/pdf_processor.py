from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from utils.system_check import get_optimal_config
from dotenv import load_dotenv
import os
import shutil
from pathlib import Path

# Load environment variables
load_dotenv()

# Ambil konfigurasi dari .env atau default
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


# 🔥 LOAD OPTIMAL CONFIGURATION
config = get_optimal_config()

# 🔥 Ambil model embedding dari config
EMBEDDING_MODEL = config['embedding_model']


def get_persist_directory(topic_name: str) -> str:
    """Mendapatkan path folder database untuk topik tertentu"""
    # Bersihkan nama topik untuk dijadikan folder name
    safe_topic = topic_name.replace(" ", "_").replace("/", "_")
    return f"./chroma_db_{safe_topic}"


def create_vector_store(pdf_path: str, topic_name: str = "default"):
    """
    Membuat vector store dengan folder TERPISAH berdasarkan topik.
    
    Args:
        pdf_path (str): Path ke file PDF
        topic_name (str): Nama topik (contoh: "skripsi_ml", "skripsi_dl")
    """
    persist_directory = get_persist_directory(topic_name)
    
    # Buat folder jika belum ada
    os.makedirs(persist_directory, exist_ok=True)
    
    print(f"📄 Membaca file PDF: {pdf_path}")
    print(f"🎯 Topic: {topic_name}")
    print(f"📁 Database: {persist_directory}")
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Batasi halaman jika RAM terbatas
    if config['max_pages'] and len(docs) > config['max_pages']:
        print(f"⚠️ Membatasi dari {len(docs)} ke {config['max_pages']} halaman")
        docs = docs[:config['max_pages']]
    
    print(f"✅ Berhasil memuat {len(docs)} halaman")
    
    print("✂️ Memecah dokumen menjadi chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap']
    )
    chunks = text_splitter.split_documents(docs)
    print(f"✅ Menjadi {len(chunks)} chunks")
    
    # Prefix untuk nomic-embed-text-v2-moe
    for chunk in chunks:
        chunk.page_content = f"search_document: {chunk.page_content}"
        chunk.metadata["topic"] = topic_name
    
    print("🔤 Membuat embeddings dengan nomic-embed-text-v2-moe...")
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )
    
    print(f"💾 Menyimpan ke Chroma database...")
    print(f"   - Topic: {topic_name}")
    print(f"   - Directory: {persist_directory}")
    print(f"   - Chunk size: {config['chunk_size']}")
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=topic_name,
        persist_directory=persist_directory
    )
    
    print("✅ Vector store berhasil dibuat!")
    return vector_store.as_retriever(search_kwargs={"k": config['k_retrieval']})


def load_vector_store(topic_name: str = "default"):
    """
    Memuat vector store berdasarkan topik dari folder yang sesuai.
    
    Args:
        topic_name (str): Nama topik yang akan dimuat
    """
    persist_directory = get_persist_directory(topic_name)
    
    if not os.path.exists(persist_directory):
        print(f"⚠️ Vector store tidak ditemukan di {persist_directory}")
        return None
    
    print(f"🔍 Memuat vector store untuk topik: {topic_name}")
    print(f"📁 Dari folder: {persist_directory}")
    
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_HOST,
    )
    
    vector_store = Chroma(
        collection_name=topic_name,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print("✅ Vector store berhasil dimuat!")
    return vector_store.as_retriever(search_kwargs={"k": config['k_retrieval']})


def delete_vector_store(topic_name: str = None):
    """
    Hapus vector store.
    - Jika topic_name diberikan: hapus folder topik tertentu
    - Jika topic_name None: hapus semua folder chroma_db_*
    """
    if topic_name:
        persist_directory = get_persist_directory(topic_name)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            print(f"🗑️ Menghapus folder: {persist_directory}")
            return True
        else:
            print(f"ℹ️ Folder tidak ditemukan: {persist_directory}")
            return False
    else:
        # Hapus semua folder chroma_db_*
        deleted = 0
        for folder in Path(".").glob("chroma_db_*"):
            if folder.is_dir():
                shutil.rmtree(folder)
                print(f"🗑️ Menghapus: {folder}")
                deleted += 1
        print(f"✅ Menghapus {deleted} folder database")
        return deleted > 0


def list_all_topics():
    """Mendapatkan daftar semua topik yang tersimpan (berdasarkan folder)"""
    topics = []
    for folder in Path(".").glob("chroma_db_*"):
        if folder.is_dir():
            # Ambil nama topik dari nama folder (hilangkan prefix "chroma_db_")
            topic_name = folder.name.replace("chroma_db_", "")
            topics.append(topic_name)
    
    # Coba baca collection dari masing-masing folder
    try:
        import chromadb
        from chromadb.config import Settings
        
        for topic in topics:
            persist_directory = get_persist_directory(topic)
            try:
                client = chromadb.PersistentClient(path=persist_directory)
                collections = client.list_collections()
                if collections:
                    # Sudah ada di topics
                    pass
            except:
                pass
    except:
        pass
    
    return topics


def topic_exists(topic_name: str) -> bool:
    """Cek apakah topik sudah memiliki database"""
    persist_directory = get_persist_directory(topic_name)
    return os.path.exists(persist_directory)