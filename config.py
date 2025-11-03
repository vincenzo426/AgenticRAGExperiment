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

# Routing keywords
DOCUMENTS_KEYWORDS = [
    "policy", "procedura", "come funziona", "processo", "normativa",
    "regolamento", "codice", "manuale", "linee guida", "dress code",
    "sicurezza", "compliance", "rimborso", "richiedere", "comunicare",
    "preavviso", "dimissioni", "onboarding", "welfare"
]

DATASET_KEYWORDS = [
    "quanti dipendenti", "statistiche", "media", "totale", "lista",
    "chi", "elenco", "confronta", "maggior", "minore", "top",
    "team", "dipartimento", "manager", "performance", "rating",
    "salario", "ferie utilizzate", "remote", "ufficio", "stipendio medio",
    "lavora", "responsabile"
]

# Hybrid: queries che potrebbero necessitare entrambi
HYBRID_INDICATORS = [
    "analisi completa", "report dettagliato", "tutto su", "panoramica",
    "spiegami e mostrami", "confronta con i dati"
]

# Dataset configuration
DATASET_PATH = "hr_dataset.csv"