import streamlit as st
import os
import sys
from pathlib import Path

# ==============================================================================
# PERBAIKAN: Paksa Python untuk mengenali root folder proyek MIRA sebagai source
# ==============================================================================
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))
# ==============================================================================

# Sekarang barulah kamu panggil import dari folder lokal kamu
from langchain_core.messages import HumanMessage
from pipeline.pdf_processor import create_vector_store
from agents.graph_mira import build_mira_graph
st.set_page_config(page_title="MIRA - Multilingual Research Assistant", layout="wide")

st.title("🤖 MIRA: Multilingual Intelligent Research Assistant")
st.subheader("Advanced Chat PDF for Academic Papers with Adaptive Translation")

# Sidebar untuk Pengaturan Dokumen dan Bahasa
with st.sidebar:
    st.header("📚 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF Jurnal (Batch/Single)", type=["pdf"])
    
    st.header("🌐 Language Settings")
    target_lang_ui = st.selectbox(
        "Bahasa Hasil Akhir Jurnal:",
        options=["Bahasa Indonesia (Default)", "English (Academic/Scopus)"]
    )
    # Mapping ke kode bahasa state MIRA
    target_lang = "id" if "Indonesia" in target_lang_ui else "en"

# Inisialisasi Session State untuk chat history dan engine
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# Memproses file yang di-upload user
if uploaded_file and st.session_state.retriever is None:
    with st.spinner("Sedang memproses dan mengindeks PDF ke FAISS Vector Store..."):
        # Simpan file sementara untuk dibaca PyPDFLoader
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Bangun Vector Store
        st.session_state.retriever = create_vector_store(temp_path)
        os.remove(temp_path) # Bersihkan file sementara
        st.success("✅ Dokumen Berhasil Diindeks! Silakan mulai bertanya.")

# Tampilan Chat Interface ala SciSpace
if st.session_state.retriever:
    # Render Histori Percakapan
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    # Input Pertanyaan User
    if user_input := st.chat_input("Tanyakan sesuatu tentang metodologi, hasil, atau temuan paper..."):
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Panggil Agen LangGraph MIRA
        mira_engine = build_mira_graph(st.session_state.retriever)
        
        with st.chat_message("assistant"):
            with st.spinner("MIRA sedang menganalisis & menyusun jawaban ilmiah..."):
                # Jalankan Graf
                inputs = {
                    "messages": [HumanMessage(content=user_input)],
                    "target_language": target_lang
                }
                output = mira_engine.invoke(inputs)
                
                # Cek apakah hasil akhir lewat translation node atau bypass
                final_answer = output.get("final_response") or output.get("raw_answer")
                st.write(final_answer)
                
        st.session_state.messages.append({"role": "assistant", "content": final_answer})
else:
    st.info("👋 Selamat datang di MIRA. Silakan upload PDF jurnal ilmiah terlebih dahulu di sidebar untuk memulai riset.")