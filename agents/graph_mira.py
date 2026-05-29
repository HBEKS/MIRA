from langgraph.graph import StateGraph, END
from agents.state import MIRAState
from agents.nodes import answer_node, translate_node, retrieve_node

def language_router(state: MIRAState):
    """Conditional Edge untuk mengecek apakah perlu translasi atau bypass."""
    # Jika bahasa asal dokumen SAMA dengan bahasa target yang diminta user, bypass langsung
    if state["source_language"] == state["target_language"]:
        return "bypass_translation"
    # Jika berbeda, arahkan ke Translation Node
    return "trigger_translation"

def build_mira_graph(retriever):
    """Membangun alur graf MIRA."""
    workflow = StateGraph(MIRAState)
    
    # Daftarkan semua Node dan bungkus retrieve_node agar bisa menerima parameter retriever
    workflow.add_node("retrieve_info", lambda state: retrieve_node(state, retriever))
    workflow.add_node("generate_raw_answer", answer_node)
    workflow.add_node("translate_academic_output", translate_node)
    
    # 2. Atur Alur Linear Awal
    workflow.set_entry_point("retrieve_info")
    workflow.add_edge("retrieve_info", "generate_raw_answer")
    
    # 3. Using Conditional Routing
    workflow.add_conditional_edges(
        "generate_raw_answer",
        language_router,
        {
            "trigger_translation": "translate_academic_output",
            "bypass_translation": END
        }
    )
    
    # 4. Sambungkan ujung Node Translasi ke SELESAI
    workflow.add_edge("translate_academic_output", END)
    
    return workflow.compile()