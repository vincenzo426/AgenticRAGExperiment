"""Configurazione del sistema Adaptive RAG"""

# Modello HuggingFace
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector store
COLLECTION_NAME = "company_kb"
TOP_K_RETRIEVAL = 3

# LLM parameters
LLM_TEMPERATURE_ROUTER = 0.1
LLM_TEMPERATURE_GENERATOR = 0.7
LLM_MAX_TOKENS_ROUTER = 100
LLM_MAX_TOKENS_GENERATOR = 512

# Chunk parameters
CHUNK_SIZE = 800
MIN_CHUNK_SIZE = 50

# RAG routing keywords
RAG_KEYWORDS = [
    "ferie", "permessi", "giorni", "benefit", "assicurazione", 
    "stipendio", "contratto", "policy", "orario", "smart working",
    "malattia", "maternità", "dimissioni", "preavviso", "bonus",
    "welfare", "formazione", "onboarding", "hr", "azienda", "ufficio",
    "rol", "congedo", "tredicesima", "tfr", "ccnl", "rimborso"
]
