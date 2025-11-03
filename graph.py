from langgraph.graph import StateGraph, END
from state import GraphState
from agents import (
    RouterAgent, 
    DocumentRetrieverAgent, 
    DatasetRetrieverAgent,
    GeneratorAgent
)


class AdaptiveRAGGraph:
    """Graph orchestrato con routing adattivo a 4 vie"""
    
    def __init__(self, hf_token: str, knowledge_base_path: str, dataset_path: str):
        print("🚀 Inizializzazione Adaptive RAG System v2")
        print("=" * 70)
        
        # Inizializza agenti
        self.router = RouterAgent(hf_token)
        self.doc_retriever = DocumentRetrieverAgent(knowledge_base_path)
        self.data_retriever = DatasetRetrieverAgent(dataset_path)
        self.generator = GeneratorAgent(hf_token)
        
        # Inizializza retriever
        self.doc_retriever.initialize()
        self.data_retriever.initialize()
        
        # Costruisci il graph
        self.graph = self._build_graph()
        
        print("=" * 70)
        print("✅ Sistema pronto!\n")
    
    def _build_graph(self) -> StateGraph:
        """Costruisce il graph con routing adattivo"""
        
        workflow = StateGraph(GraphState)
        
        # Aggiungi nodi
        workflow.add_node("router", self.router.route)
        workflow.add_node("doc_retriever", self.doc_retriever.retrieve)
        workflow.add_node("data_retriever", self.data_retriever.retrieve)
        workflow.add_node("both_retrievers", self._retrieve_both)
        workflow.add_node("generator_documents", self.generator.generate_with_documents)
        workflow.add_node("generator_data", self.generator.generate_with_data)
        workflow.add_node("generator_hybrid", self.generator.generate_hybrid)
        workflow.add_node("generator_direct", self.generator.generate_direct)
        
        # Entry point
        workflow.set_entry_point("router")
        
        # Conditional edges dal router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "direct": "generator_direct",
                "documents": "doc_retriever",
                "dataset": "data_retriever",
                "hybrid": "both_retrievers"
            }
        )
        
        # Edges dai retriever ai generator
        workflow.add_edge("doc_retriever", "generator_documents")
        workflow.add_edge("data_retriever", "generator_data")
        workflow.add_edge("both_retrievers", "generator_hybrid")
        
        # Edges finali
        workflow.add_edge("generator_documents", END)
        workflow.add_edge("generator_data", END)
        workflow.add_edge("generator_hybrid", END)
        workflow.add_edge("generator_direct", END)
        
        return workflow.compile()
    
    def _route_decision(self, state: GraphState) -> str:
        """Funzione di routing basata sulla decisione del router"""
        return state["route_decision"]
    
    def _retrieve_both(self, state: GraphState) -> GraphState:
        """Esegue entrambi i retrieval in parallelo"""
        print("🔄 Esecuzione retrieval parallelo (documenti + dati)...")
        state = self.doc_retriever.retrieve(state)
        state = self.data_retriever.retrieve(state)
        return state
    
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
            "retrieved_data": [],
            "final_answer": "",
            "confidence": 0.0,
            "sources_used": False,
            "data_used": False
        }
        
        # Esegui il graph
        final_state = self.graph.invoke(initial_state)
        
        # Mostra risultato
        print(f"\n{'='*70}")
        print(f"💡 RISPOSTA:")
        print(f"{final_state['final_answer']}")
        print(f"\n📊 Metadata:")
        print(f"   - Route: {final_state['route_decision']}")
        print(f"   - Documents used: {final_state['sources_used']}")
        print(f"   - Data used: {final_state['data_used']}")
        print(f"   - Confidence: {final_state['confidence']:.2f}")
        print(f"{'='*70}\n")
        
        return final_state
    
    def visualize_graph(self):
        """Stampa la struttura del graph"""
        print("\n📊 STRUTTURA DEL GRAPH v2:")
        print("=" * 70)
        print("""
        [START]
           ↓
        [Router] ← Analizza e decide il percorso ottimale
           ↓
        ┌──┴────────┬─────────┬────────┐
        ↓           ↓         ↓        ↓
    [Direct]  [Documents] [Dataset] [Both]
        ↓           ↓         ↓        ↓
    [Gen Direct] [Gen Doc] [Gen Data] [Gen Hybrid]
        ↓           ↓         ↓        ↓
        └───────────┴─────────┴────────┘
                    ↓
                  [END]
        
    Percorsi:
    • DIRECT → Risposta generica senza retrieval
    • DOCUMENTS → Retrieval da knowledge base testuale
    • DATASET → Retrieval da dati strutturati HR
    • HYBRID → Retrieval parallelo + risposta integrata
        """)
        print("=" * 70)