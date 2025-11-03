from langgraph.graph import StateGraph, END
from state import GraphState
from agents import RouterAgent, RetrieverAgent, GeneratorAgent


class AdaptiveRAGGraph:
    """Graph che orchestra il flusso adattivo tra gli agenti"""
    
    def __init__(self, hf_token: str, knowledge_base_path: str):
        print("🚀 Inizializzazione Adaptive RAG System")
        print("=" * 70)
        
        # Inizializza agenti
        self.router = RouterAgent(hf_token)
        self.retriever = RetrieverAgent(knowledge_base_path)
        self.generator = GeneratorAgent(hf_token)
        
        # Inizializza vector store
        self.retriever.initialize()
        
        # Costruisci il graph
        self.graph = self._build_graph()
        
        print("=" * 70)
        print("✅ Sistema pronto!\n")
    
    def _build_graph(self) -> StateGraph:
        """Costruisce il graph con i nodi e le edge"""
        
        # Crea workflow
        workflow = StateGraph(GraphState)
        
        # Aggiungi nodi
        workflow.add_node("router", self.router.route)
        workflow.add_node("retriever", self.retriever.retrieve)
        workflow.add_node("generator_rag", self.generator.generate_with_rag)
        workflow.add_node("generator_direct", self.generator.generate_direct)
        
        # Definisci entry point
        workflow.set_entry_point("router")
        
        # Conditional edge dal router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "use_rag": "retriever",
                "direct_answer": "generator_direct"
            }
        )
        
        # Edge da retriever a generator_rag
        workflow.add_edge("retriever", "generator_rag")
        
        # Edge finali
        workflow.add_edge("generator_rag", END)
        workflow.add_edge("generator_direct", END)
        
        return workflow.compile()
    
    def _route_decision(self, state: GraphState) -> str:
        """Funzione di routing basata sulla decisione del router"""
        return state["route_decision"]
    
    def query(self, question: str) -> dict:
        """Esegue una query attraverso il graph"""
        print(f"\n{'='*70}")
        print(f"❓ DOMANDA: {question}")
        print(f"{'='*70}")
        
        # Stato iniziale
        initial_state: GraphState = {
            "question": question,
            "route_decision": "",
            "retrieved_documents": [],
            "final_answer": "",
            "confidence": 0.0,
            "sources_used": False
        }
        
        # Esegui il graph
        final_state = self.graph.invoke(initial_state)
        
        # Mostra risultato
        print(f"\n{'='*70}")
        print(f"💡 RISPOSTA:")
        print(f"{final_state['final_answer']}")
        print(f"\n📊 Metadata:")
        print(f"   - Route: {final_state['route_decision']}")
        print(f"   - Sources used: {final_state['sources_used']}")
        print(f"   - Confidence: {final_state['confidence']:.2f}")
        print(f"{'='*70}\n")
        
        return final_state
    
    def visualize_graph(self):
        """Stampa la struttura del graph"""
        print("\n📊 STRUTTURA DEL GRAPH:")
        print("=" * 70)
        print("""
        [START]
           ↓
        [Router] ← Decide se usare RAG o rispondere direttamente
           ↓
        ┌──┴──┐
        ↓     ↓
    [Direct] [Retriever] ← Recupera documenti rilevanti
        ↓     ↓
        ↓  [Generator RAG] ← Genera con contesto
        ↓     ↓
        └──┬──┘
           ↓
         [END]
        """)
        print("=" * 70)