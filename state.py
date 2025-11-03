from typing import TypedDict, List, Any


class GraphState(TypedDict):
    """Stato condiviso tra i nodi del graph"""
    question: str
    route_decision: str  # "direct", "documents", "dataset", "hybrid"
    retrieved_documents: List[str]
    retrieved_data: List[Any]
    final_answer: str
    confidence: float
    sources_used: bool
    data_used: bool