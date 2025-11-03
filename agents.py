from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List
from state import GraphState
import config


class RouterAgent:
    """Agente che decide se usare RAG o rispondere direttamente"""
    
    def __init__(self, hf_token: str):
        self.client = InferenceClient(token=hf_token)
        self.model_name = config.MODEL_NAME
    
    def route(self, state: GraphState) -> GraphState:
        """Decide il routing della domanda"""
        question = state["question"]
        print(f"\n🔀 RouterAgent: Analisi domanda...")
        
        question_lower = question.lower()
        has_rag_keywords = any(keyword in question_lower for keyword in config.RAG_KEYWORDS)
        
        # Usa anche LLM per decisione più sofisticata
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Sei un router intelligente. Analizza la domanda e decidi se:
- Rispondere DIRETTAMENTE per domande generiche, saluti, richieste di chiarimenti
- Usare RAG per domande specifiche su policy aziendali, benefit, procedure HR

Rispondi SOLO con: DIRECT o RAG"""
                },
                {
                    "role": "user",
                    "content": f"Domanda: {question}"
                }
            ]
            
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=config.LLM_MAX_TOKENS_ROUTER,
                temperature=config.LLM_TEMPERATURE_ROUTER
            )
            
            result = response.choices[0].message.content
            llm_decision = "RAG" if "RAG" in result.upper() else "DIRECT"
        except Exception as e:
            print(f"⚠️ RouterAgent: LLM non disponibile, uso keyword matching")
            llm_decision = "RAG" if has_rag_keywords else "DIRECT"
        
        # Combina keyword matching e LLM
        if has_rag_keywords or llm_decision == "RAG":
            decision = "use_rag"
            print("✅ RouterAgent: Decisione → USE RAG (domanda specifica)")
        else:
            decision = "direct_answer"
            print("✅ RouterAgent: Decisione → DIRECT (domanda generica)")
        
        state["route_decision"] = decision
        return state


class RetrieverAgent:
    """Agente che recupera documenti rilevanti dalla knowledge base"""
    
    def __init__(self, knowledge_base_path: str):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.knowledge_base_path = knowledge_base_path
        
    def initialize(self):
        """Inizializza il vector store"""
        print("🔍 RetrieverAgent: Inizializzazione vector store...")
        
        with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = self._create_chunks(content)
        
        try:
            self.chroma_client.delete_collection(config.COLLECTION_NAME)
        except:
            pass
            
        self.collection = self.chroma_client.create_collection(config.COLLECTION_NAME)
        
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.encode(chunk).tolist()
            self.collection.add(
                embeddings=[embedding],
                documents=[chunk],
                ids=[f"doc_{i}"]
            )
        
        print(f"✅ RetrieverAgent: {len(chunks)} chunks indicizzati")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Crea chunks intelligenti dal testo"""
        sections = text.split('===')
        chunks = []
        
        for section in sections:
            section = section.strip()
            if len(section) > config.MIN_CHUNK_SIZE:
                # Divide sezioni lunghe in sotto-chunk
                if len(section) > config.CHUNK_SIZE:
                    lines = section.split('\n')
                    current_chunk = []
                    current_size = 0
                    
                    for line in lines:
                        if current_size + len(line) > config.CHUNK_SIZE and current_chunk:
                            chunks.append('\n'.join(current_chunk))
                            current_chunk = [line]
                            current_size = len(line)
                        else:
                            current_chunk.append(line)
                            current_size += len(line)
                    
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                else:
                    chunks.append(section)
        
        return chunks
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Recupera documenti rilevanti"""
        question = state["question"]
        print(f"🔍 RetrieverAgent: Ricerca documenti rilevanti...")
        
        query_embedding = self.embedding_model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=config.TOP_K_RETRIEVAL
        )
        
        documents = results['documents'][0] if results['documents'] else []
        state["retrieved_documents"] = documents
        state["sources_used"] = len(documents) > 0
        
        print(f"✅ RetrieverAgent: {len(documents)} documenti recuperati")
        return state


class GeneratorAgent:
    """Agente che genera la risposta finale"""
    
    def __init__(self, hf_token: str):
        self.client = InferenceClient(token=hf_token)
        self.model_name = config.MODEL_NAME
    
    def generate_with_rag(self, state: GraphState) -> GraphState:
        """Genera risposta usando RAG"""
        print("💬 GeneratorAgent: Generazione risposta con RAG...")
        
        question = state["question"]
        documents = state["retrieved_documents"]
        
        context = "\n\n".join([f"[Documento {i+1}]\n{doc}" for i, doc in enumerate(documents)])
        
        messages = [
            {
                "role": "system",
                "content": """Sei un assistente HR professionale di TechCorp Solutions. 
Rispondi basandoti ESCLUSIVAMENTE sul contesto fornito.
Sii preciso, conciso e professionale. Usa sempre l'italiano."""
            },
            {
                "role": "user",
                "content": f"""CONTESTO:
{context}

DOMANDA: {question}

Fornisci una risposta chiara e completa."""
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=config.LLM_MAX_TOKENS_GENERATOR,
                temperature=config.LLM_TEMPERATURE_GENERATOR
            )
            
            state["final_answer"] = response.choices[0].message.content.strip()
            state["confidence"] = 0.9
            print("✅ GeneratorAgent: Risposta generata con RAG")
        except Exception as e:
            state["final_answer"] = f"Errore nella generazione: {str(e)}"
            state["confidence"] = 0.0
            print(f"❌ GeneratorAgent: Errore - {str(e)}")
        
        return state
    
    def generate_direct(self, state: GraphState) -> GraphState:
        """Genera risposta diretta senza RAG"""
        print("💬 GeneratorAgent: Generazione risposta diretta...")
        
        question = state["question"]
        
        messages = [
            {
                "role": "system",
                "content": """Sei un assistente HR amichevole e professionale.
Rispondi in modo cordiale e naturale. Usa sempre l'italiano."""
            },
            {
                "role": "user",
                "content": question
            }
        ]
        
        try:
            response = self.client.chat_completion(
                messages=messages,
                model=self.model_name,
                max_tokens=config.LLM_MAX_TOKENS_GENERATOR,
                temperature=config.LLM_TEMPERATURE_GENERATOR
            )
            
            state["final_answer"] = response.choices[0].message.content.strip()
            state["confidence"] = 0.7
            state["sources_used"] = False
            print("✅ GeneratorAgent: Risposta generata direttamente")
        except Exception as e:
            state["final_answer"] = f"Errore nella generazione: {str(e)}"
            state["confidence"] = 0.0
            print(f"❌ GeneratorAgent: Errore - {str(e)}")
        
        return state