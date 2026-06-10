# utils/rag_optimizer.py
from langchain_ollama import ChatOllama
from sentence_transformers import CrossEncoder
import numpy as np

# Konfigurasi LLM untuk query rewriting
rewriter_llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0.3,
    base_url="http://127.0.0.1:11434",
)

# Load cross-encoder untuk re-ranking
try:
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    print("✅ Cross-encoder loaded for re-ranking")
except Exception as e:
    print(f"⚠️ Gagal load cross-encoder: {e}")
    reranker = None

# utils/rag_optimizer.py - Perbaiki fungsi rewrite_query


def rewrite_query(original_query: str, target_lang: str = "en") -> str:
    """
    Query Rewriting: Memperbaiki query untuk retrieval yang lebih baik.
    """
    if target_lang == "id":
        prompt = f"""
        Anda adalah asisten riset akademik. Ubah pertanyaan berikut menjadi query pencarian yang lebih efektif.

        Aturan:
        1. Hanya keluarkan query hasil rewrite, tanpa penjelasan tambahan
        2. Pertahankan makna asli pertanyaan
        3. Gunakan kata kunci yang relevan
        4. Buat lebih spesifik dan ringkas

        Pertanyaan asli: "{original_query}"

        Query hasil rewrite:
        """
    else:
        prompt = f"""
        You are an academic research assistant. Rewrite the following question into a more effective search query.

        Rules:
        1. Output ONLY the rewritten query, no additional text or explanation
        2. Preserve the original meaning
        3. Use relevant keywords
        4. Make it more specific and concise

        Original question: "{original_query}"

        Rewritten query:
        """

    try:
        response = rewriter_llm.invoke(prompt)
        rewritten = response.content.strip()

        # 🔥 CLEANUP: Hapus kalimat pembuka yang tidak perlu
        # Hapus "Here is a rewritten version..." jika ada
        unwanted_prefixes = [
            "Here is a rewritten version",
            "Here is the rewritten query",
            "Rewritten query:",
            "Here's",
            "The rewritten query is"
        ]

        for prefix in unwanted_prefixes:
            if rewritten.lower().startswith(prefix.lower()):
                # Ambil setelah prefix
                rewritten = rewritten.split(
                    ":", 1)[-1].strip() if ":" in rewritten else rewritten
                break

        # Hapus kutipan di awal/akhir jika ada
        rewritten = rewritten.strip('"\'')

        # Jika masih kosong atau terlalu pendek, gunakan original
        if len(rewritten) < 5:
            print(f"⚠️ Query rewriting returned empty, using original")
            return original_query

        print(
            f"✍️ Query rewriting: '{original_query[:40]}...' → '{rewritten[:60]}...'")
        return rewritten

    except Exception as e:
        print(f"⚠️ Query rewriting failed: {e}, using original")
        return original_query


def rerank_chunks(chunks: list, query: str, top_k: int = 5) -> list:
    """Re-ranking: Mengurutkan ulang chunks berdasarkan relevansi."""
    if not chunks or reranker is None:
        return chunks[:top_k] if chunks else []

    try:
        pairs = [(query, chunk.page_content) for chunk in chunks]
        scores = reranker.predict(pairs)

        chunk_score_pairs = list(zip(chunks, scores))
        chunk_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_chunks = [chunk for chunk, score in chunk_score_pairs[:top_k]]

        print(
            f"📊 Re-ranking: {len(chunks)} chunks → top {len(reranked_chunks)} chunks")
        return reranked_chunks

    except Exception as e:
        print(f"⚠️ Re-ranking failed: {e}")
        return chunks[:top_k]


def process_query_with_optimization(original_query: str, retriever, target_lang: str = "en",
                                    use_rewriting: bool = True, use_reranking: bool = True):
    """Proses query dengan optimasi RAG lengkap."""

    # Step 1: Query Rewriting
    if use_rewriting:
        search_query = rewrite_query(original_query, target_lang)
    else:
        search_query = original_query

    # Step 2: Initial Retrieval (ambil lebih banyak untuk di-rerank)
    initial_k = 20 if use_reranking else 6
    query_with_prefix = f"search_query: {search_query}"

    print(f"🔍 Initial retrieval: mengambil {initial_k} chunks...")
    relevant_docs = retriever.invoke(query_with_prefix)

    # Step 3: Re-ranking
    if use_reranking and len(relevant_docs) > 5:
        top_chunks = rerank_chunks(relevant_docs, original_query, top_k=5)
    else:
        top_chunks = relevant_docs[:5]

    # Step 4: Build context
    context_text = "\n\n".join([doc.page_content for doc in top_chunks])
    clean_context = context_text.replace("search_document: ", "")

    print(f"✅ Final: {len(top_chunks)} chunks setelah optimasi")

    return clean_context, search_query, top_chunks
