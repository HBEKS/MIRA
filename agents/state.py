from typing import List, TypedDict
from langchain_core.messages import BaseMessage

class MIRAState(TypedDict):
    messages: List[BaseMessage]   # Histori percakapan
    pdf_context: str              # Hasil retrieval dari ChromaDB
    source_language: str          # Bahasa dokumen asal ('en' atau 'id')
    target_language: str          # Bahasa output pilihan user ('en' atau 'id')
    raw_answer: str               # Jawaban murni dalam bahasa asli dokumen
    final_response: str           # Jawaban akhir setelah penyesuaian bahasa