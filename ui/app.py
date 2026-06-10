import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
import json
from datetime import datetime

root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

# Import modul yang diperlukan (with try catch error handling)
try:
    from langchain_core.messages import HumanMessage
    from pipeline.pdf_processor import (
        create_vector_store, 
        load_vector_store, 
        delete_vector_store, 
        list_all_topics,
        topic_exists
    )
    from agents.graph_mira import build_mira_graph
    from utils.system_check import get_system_specs, get_optimal_config
except ImportError as e:
    st.error(f"❌ Gagal mengimpor modul: {e}")
    st.info("Pastikan semua package sudah terinstall: `uv add langchain langchain-ollama streamlit psutil`")
    sys.exit(1)

# Streamlit page configuration
st.set_page_config(
    page_title="MIRA - Multilingual Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Chat History Configuration
CHAT_HISTORY_DIR = Path("./chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

def save_chat_to_disk(chat_id: str, messages: list, pdf_name: str, topic: str):
    """Menyimpan chat history ke file JSON"""
    try:
        history_file = CHAT_HISTORY_DIR / f"{chat_id}.json"
        data = {
            "chat_id": chat_id,
            "messages": messages,
            "pdf_name": pdf_name,
            "topic": topic,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "message_count": len(messages) // 2
        }
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ Gagal menyimpan chat: {e}")
        return False

def load_chat_from_disk(chat_id: str):
    """Memuat chat history dari file JSON"""
    try:
        history_file = CHAT_HISTORY_DIR / f"{chat_id}.json"
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"❌ Gagal memuat chat: {e}")
    return None

def list_all_chats():
    """Mendapatkan daftar semua chat yang tersimpan"""
    chats = []
    for history_file in CHAT_HISTORY_DIR.glob("*.json"):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                chats.append({
                    "chat_id": data["chat_id"],
                    "created_at": data.get("created_at", ""),
                    "last_updated": data.get("last_updated", ""),
                    "message_count": data.get("message_count", 0),
                    "pdf_name": data.get("pdf_name", ""),
                    "topic": data.get("topic", "default")
                })
        except Exception as e:
            print(f"⚠️ Error membaca {history_file}: {e}")
    
    chats.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
    return chats

def delete_chat_from_disk(chat_id: str):
    """Menghapus file chat history"""
    try:
        history_file = CHAT_HISTORY_DIR / f"{chat_id}.json"
        if history_file.exists():
            history_file.unlink()
            return True
    except Exception as e:
        print(f"❌ Gagal menghapus chat: {e}")
    return False

# Session State Initialization
def init_session_state():
    """Inisialisasi session state dengan nilai default yang aman"""
    defaults = {
        "messages": [],
        "retriever": None,
        "current_pdf_name": None,
        "system_info_shown": False,
        "current_chat_id": None,
        "current_topic": "skripsi_ml",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Chat Management Function
def start_new_chat():
    """Memulai chat baru - SAVE dulu chat lama"""
    if st.session_state.current_chat_id and st.session_state.messages:
        save_chat_to_disk(
            st.session_state.current_chat_id,
            st.session_state.messages,
            st.session_state.current_pdf_name or "",
            st.session_state.current_topic
        )
    
    st.session_state.messages = []
    st.session_state.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.retriever = None
    st.session_state.current_pdf_name = None
    
    save_chat_to_disk(
        st.session_state.current_chat_id,
        [],
        "",
        st.session_state.current_topic
    )

def load_chat(chat_id: str):
    """Memuat chat history - SAVE dulu chat aktif sebelum pindah"""
    if st.session_state.current_chat_id and st.session_state.messages:
        save_chat_to_disk(
            st.session_state.current_chat_id,
            st.session_state.messages,
            st.session_state.current_pdf_name or "",
            st.session_state.current_topic
        )
    
    chat_data = load_chat_from_disk(chat_id)
    if chat_data:
        st.session_state.messages = chat_data.get("messages", [])
        st.session_state.current_chat_id = chat_id
        st.session_state.current_topic = chat_data.get("topic", "skripsi_ml")
        st.session_state.current_pdf_name = chat_data.get("pdf_name", "")
        
        # Load vector store berdasarkan topik yang tersimpan
        if st.session_state.current_pdf_name:
            st.session_state.retriever = load_vector_store(st.session_state.current_topic)

def delete_chat(chat_id: str):
    """Menghapus chat dari disk"""
    # Ambil data chat sebelum dihapus
    chat_data = load_chat_from_disk(chat_id)
    topic_to_delete = chat_data.get("topic", "") if chat_data else ""
    
    # Hapus file chat
    delete_chat_from_disk(chat_id)
    
    # Hapus folder database jika tidak ada chat lain yang menggunakan topik yang sama
    if topic_to_delete:
        other_chats_with_same_topic = False
        for chat in list_all_chats():
            if chat.get("topic") == topic_to_delete and chat["chat_id"] != chat_id:
                other_chats_with_same_topic = True
                break
        
        if not other_chats_with_same_topic:
            delete_vector_store(topic_to_delete)
    
    # Jika yang dihapus adalah chat yang aktif
    if st.session_state.current_chat_id == chat_id:
        remaining_chats = list_all_chats()
        if remaining_chats:
            load_chat(remaining_chats[0]["chat_id"])
        else:
            start_new_chat()
    
    st.rerun()

def save_current_chat():
    """Menyimpan chat yang sedang aktif"""
    if st.session_state.current_chat_id:
        save_chat_to_disk(
            st.session_state.current_chat_id,
            st.session_state.messages,
            st.session_state.current_pdf_name or "",
            st.session_state.current_topic
        )

# Load chat terakhir saat startup
saved_chats = list_all_chats()
if saved_chats and st.session_state.current_chat_id is None:
    latest_chat = saved_chats[0]
    chat_data = load_chat_from_disk(latest_chat["chat_id"])
    if chat_data:
        st.session_state.current_chat_id = chat_data["chat_id"]
        st.session_state.messages = chat_data.get("messages", [])
        st.session_state.current_topic = chat_data.get("topic", "skripsi_ml")
        st.session_state.current_pdf_name = chat_data.get("pdf_name", "")
        
        if st.session_state.current_pdf_name:
            st.session_state.retriever = load_vector_store(st.session_state.current_topic)


# System and configuration info - Sidebar
with st.sidebar:
    # CHAT MANAGEMENT
    st.markdown("## 💬 Chat Management")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            start_new_chat()
            st.rerun()
    
    saved_chats = list_all_chats()
    if saved_chats:
        st.markdown("### 📜 History")
        for chat in saved_chats[:5]:
            chat_id = chat["chat_id"]
            chat_time = chat.get("created_at", "")[:16].replace("T", " ")
            msg_count = chat.get("message_count", 0)
            topic = chat.get("topic", "default")
            is_active = (chat_id == st.session_state.current_chat_id)
            
            # Icon berdasarkan topik
            if "ml" in topic.lower() or "machine" in topic.lower():
                topic_icon = "🧠"
            elif "dl" in topic.lower() or "deep" in topic.lower():
                topic_icon = "🔬"
            elif "cv" in topic.lower() or "vision" in topic.lower():
                topic_icon = "👁️"
            else:
                topic_icon = "📝"
            
            label = f"{'🔵 ' if is_active else '📝 '}{topic_icon} {chat_time} ({msg_count})"
            
            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(label, key=f"load_{chat_id}", use_container_width=True):
                    load_chat(chat_id)
                    st.rerun()
            with col_b:
                if st.button("🗑️", key=f"del_{chat_id}", help="Delete"):
                    delete_chat(chat_id)
                    st.rerun()
    
    st.markdown("---")

    # Topic Selection
    st.markdown("## 🎯 Research Topic")
    
    topic_options = {
        "🧠 Machine Learning": "skripsi_ml",
        "🔬 Deep Learning": "skripsi_dl",
        "👁️ Computer Vision": "skripsi_cv",
        "💬 NLP": "skripsi_nlp",
    }
    
    selected_topic_label = st.selectbox(
        "Pilih Topik Penelitian:",
        options=list(topic_options.keys()),
        index=0,
        help="Pilih topik agar database vector terisolasi per penelitian"
    )
    
    st.session_state.current_topic = topic_options[selected_topic_label]
    
    # Tampilkan topik yang tersedia
    existing_topics = list_all_topics()
    if existing_topics:
        with st.expander("📁 Existing Databases", expanded=False):
            for t in existing_topics:
                st.caption(f"- {t}")
    
    st.markdown("---")
    
    # Document Ingestion (Upload PDF)
    st.markdown("## 📚 Document Ingestion")
    
    uploaded_file = st.file_uploader(
        "Upload PDF Jurnal", 
        type=["pdf"],
        help=f"Upload PDF untuk topik: {selected_topic_label}"
    )
    
    if st.session_state.current_pdf_name:
        st.info(f"📄 **Active:** `{st.session_state.current_pdf_name}`")
        st.caption(f"🎯 **Topic:** {selected_topic_label}")
    
    # Tampilkan info sistem
    if not st.session_state.system_info_shown:
        with st.expander("🖥️ System Information", expanded=False):
            specs = get_system_specs()
            config = get_optimal_config()
            
            st.markdown(f"""
            **RAM:** {specs['total_ram_gb']:.0f} GB  
            **CPU:** {specs['cpu_cores']} cores  
            **GPU:** {'✅' if specs['gpu_available'] else '❌'}  
            **LLM Model:** `{config['llm_model']}`  
            **Mode:** {'⚡ High Performance' if specs['total_ram_gb'] >= 32 else '📱 Balanced' if specs['total_ram_gb'] >= 16 else '🐢 Lite Mode'}
            """)
        st.session_state.system_info_shown = True
    
    st.markdown("---")
    st.markdown("## 🌐 Language Settings")
    
    target_lang_ui = st.radio(
        "Output Language:",
        options=["Bahasa Indonesia", "English"],
        index=0,
        horizontal=True,
    )
    target_lang = "id" if target_lang_ui == "Bahasa Indonesia" else "en"
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.messages = []
        save_current_chat()
        st.rerun()
    
    if st.session_state.retriever is not None:
        if st.button("🔄 Reset Document", use_container_width=True):
            try:
                delete_vector_store(st.session_state.current_topic)
                st.session_state.retriever = None
                st.session_state.current_pdf_name = None
                st.session_state.messages = []
                save_current_chat()
                st.success(f"✅ Dokumen untuk topik {selected_topic_label} berhasil di-reset!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Gagal reset: {e}")

# MAIN CONTENT - TITLE & HEADER
st.title("🤖 MIRA: Multilingual Intelligent Research Assistant")
st.caption("Advanced Chat PDF for Academic Papers with Adaptive Translation")

# Tampilkan status
if st.session_state.current_pdf_name:
    st.info(f"📄 **Active Document:** `{st.session_state.current_pdf_name}` | 🎯 **Topic:** {selected_topic_label}")
else:
    st.info(f"👋 **Selamat datang di MIRA!**\n\nPilih topik penelitian, lalu upload PDF jurnal ilmiah di sidebar untuk memulai riset. Database vector akan terisolasi per topik!")

# PROSES UPLOAD PDF
if uploaded_file is not None:
    if st.session_state.current_pdf_name != uploaded_file.name:
        try:
            with st.spinner(f"📑 Sedang memproses PDF untuk topik {selected_topic_label}..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    temp_path = tmp_file.name
                
                # Make Vector Store with specific topic (Seperate folder for each topic)
                st.session_state.retriever = create_vector_store(
                    temp_path, 
                    topic_name=st.session_state.current_topic
                )
                
                st.session_state.current_pdf_name = uploaded_file.name
                st.session_state.messages = []
                save_current_chat()
                
                os.unlink(temp_path)
                
                st.success(f"✅ Dokumen **{uploaded_file.name}** berhasil diindeks untuk topik {selected_topic_label}!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Gagal memproses PDF: {str(e)}")
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
    else:
        st.info(f"📄 Dokumen **{uploaded_file.name}** sudah aktif. Lanjutkan bertanya.")

# CHAT INTERFACE
if st.session_state.retriever:
    # Tampilkan histori chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Input chat
    if user_input := st.chat_input("Tanyakan sesuatu tentang metodologi, hasil, atau temuan paper..."):
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        try:
            mira_engine = build_mira_graph(st.session_state.retriever)
            
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "target_language": target_lang
            }
            
            with st.chat_message("assistant"):
                with st.spinner("🔬 MIRA sedang menganalisis & menyusun jawaban ilmiah..."):
                    output = mira_engine.invoke(inputs)
                    final_answer = output.get("final_response") or output.get("raw_answer", "Maaf, tidak dapat memproses permintaan.")
                    st.markdown(final_answer)
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            save_current_chat()
            
        except Exception as e:
            st.error(f"❌ Error saat memproses pertanyaan: {str(e)}")
            st.info("Silakan coba lagi atau upload ulang dokumen.")
            
else:
    # Tampilan awal ketika belum ada PDF
    if not uploaded_file:
        with st.expander("📖 Contoh Pertanyaan yang Bisa Diajukan"):
            st.markdown("""
            - **Metodologi:** *"Apa metode yang digunakan dalam penelitian ini?"*
            - **Hasil:** *"Apa temuan utama dari paper ini?"*
            - **Evaluasi:** *"Bagaimana performa model dibandingkan dengan baseline?"*
            - **Kesimpulan:** *"Apa kontribusi utama dari penelitian ini?"*
            """)