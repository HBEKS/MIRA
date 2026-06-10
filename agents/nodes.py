# agents/nodes.py
from langchain_ollama import ChatOllama
from agents.state import MIRAState
from langdetect import detect, LangDetectException
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.rag_optimizer import process_query_with_optimization  # 🔥 TAMBAHAN
import sys
import os

# ==============================================================================
# KONFIGURASI OLLAMA LLM
# ==============================================================================

# Pilih model LLM yang sudah di-pull
# Rekomendasi: "llama3.2:3b" (akurat) atau "gemma2:2b" (lebih ringan)
LLM_MODEL = "llama3.2:3b"  # Ganti sesuai model yang Anda pull

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    base_url="http://127.0.0.1:11434",  # 🔥 Ganti ke 127.0.0.1 lebih stabil
    num_predict=2048,  # Maksimal token output
)

print(f"✅ MIRA menggunakan LLM: {LLM_MODEL}")


# ==============================================================================
# RETRY MECHANISM (untuk menghandle error sementara)
# ==============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def invoke_llm_with_retry(messages):
    """Invoke LLM dengan retry mechanism untuk chat messages"""
    return llm.invoke(messages)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True
)
def invoke_llm_prompt_with_retry(prompt: str):
    """Invoke LLM dengan retry mechanism untuk prompt string"""
    return llm.invoke(prompt)


# ==============================================================================
# NODE 1: RETRIEVE NODE (DENGAN OPTIMASI RAG)
# ==============================================================================

def retrieve_node(state: MIRAState, retriever):
    """
    Node 1: Mengambil dokumen relevan dari vector store.
    
    🔥 KRUSIAL untuk nomic-embed-text-v2-moe:
    - Query harus diberi prefix "search_query: "
    - Dokumen di database sudah diberi prefix "search_document: "
    
    🔥 OPTIMASI: Query Rewriting + Re-ranking (Advanced RAG)
    """
    user_query = state["messages"][-1].content
    target_lang = state.get("target_language", "en")
    
    # ==========================================================================
    # 🔥 TAMBAHAN: Gunakan optimasi RAG (Query Rewriting + Re-ranking)
    # ==========================================================================
    USE_OPTIMIZATION = True  # Set ke False jika ingin pakai naive RAG
    
    if USE_OPTIMIZATION:
        print("="*50)
        print("🚀 RAG OPTIMIZATION PIPELINE (Query Rewriting + Re-ranking)")
        print("="*50)
        
        clean_context, used_query, top_chunks = process_query_with_optimization(
            original_query=user_query,
            retriever=retriever,
            target_lang=target_lang,
            use_rewriting=True,
            use_reranking=True
        )
        
        print(f"✅ Retrieved {len(top_chunks)} optimized chunks")
        print("="*50)
        
        # Deteksi bahasa dari context (untuk kompatibilitas dengan translate_node)
        detected_lang = "en"
        if clean_context.strip():
            try:
                sample_text = clean_context[:300]
                detected_lang = detect(sample_text)
                if detected_lang not in ["en", "id"]:
                    detected_lang = "en"
            except LangDetectException:
                detected_lang = "en"
        
        print(f"✅ Retrieved {len(top_chunks)} chunks, detected language: {detected_lang}")
        
        return {
            "pdf_context": clean_context,
            "source_language": detected_lang
        }
    
    else:
        # ======================================================================
        # KODE LAMA (NAIVE RAG) - TETAP DI SINI
        # ======================================================================
        
        # 🔥 PENTING: Prefix query untuk nomic-embed-text-v2-moe
        query_with_prefix = f"search_query: {user_query}"
        
        print(f"🔍 Melakukan retrieval dengan query: {query_with_prefix[:100]}...")
        
        # Jalankan retrieval dengan query yang sudah diberi prefix
        relevant_docs = retriever.invoke(query_with_prefix)
        
        # Gabungkan semua dokumen yang relevan
        context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Deteksi bahasa dari context (untuk instruksi bahasa di answer_node)
        detected_lang = "en"
        if context_text.strip():
            try:
                sample_text = context_text[:300]
                detected_lang = detect(sample_text)
                if detected_lang not in ["en", "id"]:
                    detected_lang = "en"
            except LangDetectException:
                detected_lang = "en"
        
        # 🔥 Hapus prefix "search_document: " dari context sebelum dikirim ke LLM
        clean_context = context_text.replace("search_document: ", "")
        
        print(f"✅ Retrieved {len(relevant_docs)} chunks, detected language: {detected_lang}")
        
        return {
            "pdf_context": clean_context,
            "source_language": detected_lang
        }


# ==============================================================================
# NODE 2: ANSWER NODE (TIDAK BERUBAH)
# ==============================================================================

def answer_node(state: MIRAState):
    """
    Node 2: Menjawab berdasarkan teks PDF (Prioritas Utama) 
            dan Pengetahuan General (Pendukung).
    """
    context = state["pdf_context"]
    user_query = state["messages"][-1].content
    source_lang = state["source_language"]
    
    # Instruksi bahasa berdasarkan deteksi otomatis
    if source_lang == "en":
        lang_instruction = "Respond in formal, structured Academic English."
    else:
        lang_instruction = "Respond in formal, structured academic Indonesian (Bahasa Indonesia ilmiah baku)."
    
    system_prompt = (
        "You are MIRA, an elite academic research assistant. Your primary task is to answer the user's question "
        "based on the provided scientific context from the uploaded PDF. "
        "\n\nCRITICAL INSTRUCTIONS:\n"
        "1. PRIORITIZE the provided context. If the answer is directly available in the text, use it as your main reference.\n"
        "2. FLEXIBLE EXPANSION: If the user asks about general concepts, terminologies, or background mathematics related "
        "to the context (but not deeply explained in the PDF), you are FULLY ALLOWED and encouraged to use your internal "
        "knowledge base to provide a comprehensive, educational explanation.\n"
        "3. Clear Boundary: If the question is completely unrelated to the PDF or any computer vision/AI domain at all, "
        "politely guide the user back to the scope of the document.\n\n"
        f"{lang_instruction}\n\n"
        f"Provided PDF Context:\n{context}"
    )
    
    try:
        print("💡 MIRA sedang menyusun jawaban...")
        response = invoke_llm_with_retry([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ])
        print("✅ Jawaban berhasil disusun")
        return {"raw_answer": response.content}
        
    except Exception as e:
        print(f"❌ Error di answer_node: {e}")
        return {
            "raw_answer": f"Maaf, terjadi kesalahan teknis saat menyusun jawaban. Silakan coba lagi. Error: {str(e)}"
        }


# ==============================================================================
# NODE 3: TRANSLATE NODE (TIDAK BERUBAH)
# ==============================================================================

def translate_node(state: MIRAState):
    """
    Node 3: Mengubah jawaban sains ke bahasa target dengan tata bahasa akademis tinggi.
    """
    raw_answer = state["raw_answer"]
    target_lang = state["target_language"]
    
    # Pilih prompt berdasarkan target bahasa
    if target_lang == "id":
        prompt = (
            "Bertindaklah sebagai penerjemah jurnal ilmiah profesional. Terjemahkan teks akademik "
            "berbahasa Inggris berikut ke dalam Bahasa Indonesia akademik yang formal, natural, "
            "dan mudah dipahami oleh dosen serta mahasiswa tingkat akhir. "
            "Kepatuhan Ketat: Pertahankan istilah teknis, singkatan ilmiah, nama algoritma, atau rumus matematika "
            f"aslinya jika tidak memiliki padanan baku yang tepat dalam Bahasa Indonesia:\n\n{raw_answer}"
        )
    else:
        prompt = (
            "Act as a professional scientific editor for high-impact journals. Transform the following "
            "Indonesian academic text into high-quality, professional International Academic English. "
            "Ensure precise lexical choices, formal syntax, and standard scientific terminology suitable "
            f"for Scopus or Web of Science indexed publications:\n\n{raw_answer}"
        )
    
    try:
        print(f"🌐 MIRA sedang menerjemahkan ke bahasa: {target_lang.upper()}...")
        response = invoke_llm_prompt_with_retry(prompt)
        print("✅ Terjemahan selesai")
        return {"final_response": response.content}
        
    except Exception as e:
        print(f"❌ Error di translate_node: {e}")
        return {
            "final_response": f"Maaf, terjadi kesalahan dalam terjemahan. Silakan coba lagi. Error: {str(e)}"
        }