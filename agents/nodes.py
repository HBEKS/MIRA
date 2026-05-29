from langchain_openai import ChatOpenAI
from agents.state import MIRAState
from langdetect import detect, LangDetectException
from config.settings import settings

# Konfigurasi LLM dengan OpenRouter
llm = ChatOpenAI(
    model="deepseek/deepseek-v4-flash-20260423:free", # Model gratis yang tersedia di OpenRouter
    api_key=settings.OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    default_headers={
        "HTTP-Referer": "https://github.com/danielwuliutomo/MIRA",
        "X-Title": "MIRA Assistant"
    }
)

def retrieve_node(state: MIRAState, retriever):
    """Node 1: Mengambil dokumen relevan dari FAISS dan mendeteksi bahasa asli jurnal."""
    user_query = state["messages"][-1].content
    
    # Menjalankan pemanggilan invoke standar versi terbaru
    relevant_docs = retriever.invoke(user_query)
    
    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    detected_lang = "en"
    if context_text.strip():
        try:
            sample_text = context_text[:300]
            detected_lang = detect(sample_text)
            if detected_lang not in ["en", "id"]:
                detected_lang = "en"
        except LangDetectException:
            detected_lang = "en"

    return {"pdf_context": context_text, "source_language": detected_lang}

def answer_node(state: MIRAState):
    """Node 2: Menjawab berdasarkan teks PDF (Prioritas Utama) dan Pengetahuan General (Pendukung)."""
    context = state["pdf_context"]
    user_query = state["messages"][-1].content
    source_lang = state["source_language"]
    
    if source_lang == "en":
        lang_instruction = "Respond in formal, structured Academic English."
    else:
        lang_instruction = "Respond in formal, structured academic Indonesian (Bahasa Indonesia ilmiah baku)."
    
    # Mengizinkan AI memakai general knowledge jika data di PDF kurang detail
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
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ])
    
    return {"raw_answer": response.content}

def translate_node(state: MIRAState):
    """Node 3: Mengubah jawaban sains ke bahasa luaran target dengan tata bahasa akademis tinggi."""
    raw_answer = state["raw_answer"]
    target_lang = state["target_language"]
    
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
        
    response = llm.invoke(prompt)
    return {"final_response": response.content}