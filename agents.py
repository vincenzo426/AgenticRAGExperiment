from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from typing import List
from state import GraphState
import config


class RouterAgent:
    """Agente che decide il routing: direct, documents, dataset o hybrid"""
    
    def __init__(self, hf_token: str):
        self.client = InferenceClient(token=hf_token)
        self.model_name = config.MODEL_NAME
    
    def route(self, state: GraphState) -> GraphState:
        """Decide il routing della domanda"""
        question = state["question"]
        print(f"\n🔀 RouterAgent: Analisi domanda...")
        
        question_lower = question.lower()
        
        # Keyword matching
        has_document_keywords = any(kw in question_lower for kw in config.DOCUMENTS_KEYWORDS)
        has_dataset_keywords = any(kw in question_lower for kw in config.DATASET_KEYWORDS)
        has_hybrid_indicators = any(ind in question_lower for ind in config.HYBRID_INDICATORS)
        
        # LLM-based routing
        try:
            messages = [
                {
                    "role": "system",
                    "content": """Sei un router intelligente. Analizza la domanda e decidi:
- DIRECT: domande generiche, saluti, ringraziamenti
- DOCUMENTS: domande su policy, procedure, normative aziendali, benefit
- DATASET: domande su dati specifici di dipendenti, statistiche, numeri, metriche
- HYBRID: domande che richiedono sia policy che dati (es. "mostrami chi ha usato più ferie e qual è la policy")

Rispondi SOLO con: DIRECT, DOCUMENTS, DATASET o HYBRID"""
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
            
            result = response.choices[0].message.content.upper()
            
            if "HYBRID" in result:
                llm_decision = "hybrid"
            elif "DATASET" in result:
                llm_decision = "dataset"
            elif "DOCUMENTS" in result:
                llm_decision = "documents"
            else:
                llm_decision = "direct"
                
        except Exception as e:
            print(f"⚠️ RouterAgent: LLM non disponibile, uso keyword matching")
            # Fallback a keyword matching
            if has_hybrid_indicators or (has_document_keywords and has_dataset_keywords):
                llm_decision = "hybrid"
            elif has_dataset_keywords:
                llm_decision = "dataset"
            elif has_document_keywords:
                llm_decision = "documents"
            else:
                llm_decision = "direct"
        
        # Decisione finale combinando keywords e LLM
        if has_hybrid_indicators or (has_document_keywords and has_dataset_keywords):
            decision = "hybrid"
            print("✅ RouterAgent: Decisione → HYBRID (documenti + dataset)")
        elif llm_decision == "dataset" or has_dataset_keywords:
            decision = "dataset"
            print("✅ RouterAgent: Decisione → DATASET (dati strutturati)")
        elif llm_decision == "documents" or has_document_keywords:
            decision = "documents"
            print("✅ RouterAgent: Decisione → DOCUMENTS (knowledge base)")
        else:
            decision = "direct"
            print("✅ RouterAgent: Decisione → DIRECT (risposta generica)")
        
        state["route_decision"] = decision
        return state


class DocumentRetrieverAgent:
    """Agente che recupera documenti dalla knowledge base"""
    
    def __init__(self, knowledge_base_path: str):
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
        self.chroma_client = chromadb.Client()
        self.collection = None
        self.knowledge_base_path = knowledge_base_path
        
    def initialize(self):
        """Inizializza il vector store"""
        print("📄 DocumentRetriever: Inizializzazione vector store...")
        
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
        
        print(f"✅ DocumentRetriever: {len(chunks)} chunks indicizzati")
    
    def _create_chunks(self, text: str) -> List[str]:
        """Crea chunks intelligenti dal testo"""
        sections = text.split('===')
        chunks = []
        
        for section in sections:
            section = section.strip()
            if len(section) > config.MIN_CHUNK_SIZE:
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
        print(f"📄 DocumentRetriever: Ricerca documenti...")
        
        query_embedding = self.embedding_model.encode(question).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=config.TOP_K_RETRIEVAL
        )
        
        documents = results['documents'][0] if results['documents'] else []
        state["retrieved_documents"] = documents
        state["sources_used"] = len(documents) > 0
        
        print(f"✅ DocumentRetriever: {len(documents)} documenti recuperati")
        return state


class DatasetRetrieverAgent:
    """Agente che recupera dati dal dataset HR"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = None
        
    def initialize(self):
        """Carica il dataset"""
        print("📊 DatasetRetriever: Caricamento dataset...")
        self.df = pd.read_csv(self.dataset_path)
        print(f"✅ DatasetRetriever: {len(self.df)} record caricati")
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Recupera dati rilevanti dal dataset"""
        question = state["question"]
        print(f"📊 DatasetRetriever: Analisi query sui dati...")
        
        question_lower = question.lower()
        results = []
        
        # Query specifiche su dipendenti
        if "quanti dipendenti" in question_lower:
            total = len(self.df)
            by_dept = self.df['department'].value_counts().to_dict()
            results.append({
                "type": "count",
                "total_employees": total,
                "by_department": by_dept
            })
        
        # Statistiche ferie
        if "ferie" in question_lower and ("media" in question_lower or "statistiche" in question_lower):
            avg_used = self.df['vacation_days_used'].mean()
            max_used = self.df['vacation_days_used'].max()
            min_used = self.df['vacation_days_used'].min()
            results.append({
                "type": "vacation_stats",
                "average": round(avg_used, 1),
                "max": int(max_used),
                "min": int(min_used)
            })
        
        # Top performer per ferie utilizzate
        if "più ferie" in question_lower or "maggior" in question_lower:
            top_vacation = self.df.nlargest(5, 'vacation_days_used')[
                ['full_name', 'vacation_days_used', 'department']
            ].to_dict('records')
            results.append({
                "type": "top_vacation",
                "employees": top_vacation
            })
        
        # Remote working
        if "remote" in question_lower or "smart working" in question_lower:
            avg_remote = self.df['remote_days_per_week'].mean()
            by_dept = self.df.groupby('department')['remote_days_per_week'].mean().to_dict()
            results.append({
                "type": "remote_stats",
                "average_days": round(avg_remote, 1),
                "by_department": {k: round(v, 1) for k, v in by_dept.items()}
            })
        
        # Manager e team
        if "team" in question_lower or "manager" in question_lower:
            managers = self.df[self.df['manager'].notna()].groupby('manager').size().to_dict()
            results.append({
                "type": "team_structure",
                "teams": managers
            })
        
        # Performance
        if "performance" in question_lower or "rating" in question_lower:
            avg_rating = self.df['performance_rating'].mean()
            top_performers = self.df.nlargest(5, 'performance_rating')[
                ['full_name', 'performance_rating', 'role']
            ].to_dict('records')
            results.append({
                "type": "performance_stats",
                "average_rating": round(avg_rating, 2),
                "top_performers": top_performers
            })
        
        # Stipendi
        if "stipendio" in question_lower or "salario" in question_lower:
            avg_salary = self.df['salary'].mean()
            by_dept = self.df.groupby('department')['salary'].mean().to_dict()
            results.append({
                "type": "salary_stats",
                "average_salary": round(avg_salary, 0),
                "by_department": {k: round(v, 0) for k, v in by_dept.items()}
            })
        
        # Chi lavora in una location specifica
        if any(loc in question_lower for loc in ["milano", "roma", "bologna"]):
            for loc in ["Milano", "Roma", "Bologna"]:
                if loc.lower() in question_lower:
                    emp_list = self.df[self.df['location'] == loc][
                        ['full_name', 'role', 'department']
                    ].to_dict('records')
                    results.append({
                        "type": "location_employees",
                        "location": loc,
                        "count": len(emp_list),
                        "employees": emp_list[:10]  # Primi 10
                    })
        
        # Fallback: info generali
        if not results:
            results.append({
                "type": "general",
                "total_employees": len(self.df),
                "departments": self.df['department'].unique().tolist(),
                "locations": self.df['location'].unique().tolist()
            })
        
        state["retrieved_data"] = results
        state["data_used"] = len(results) > 0
        
        print(f"✅ DatasetRetriever: {len(results)} risultati trovati")
        return state


class GeneratorAgent:
    """Agente che genera la risposta finale"""
    
    def __init__(self, hf_token: str):
        self.client = InferenceClient(token=hf_token)
        self.model_name = config.MODEL_NAME
    
    def generate_with_documents(self, state: GraphState) -> GraphState:
        """Genera risposta usando i documenti"""
        print("💬 GeneratorAgent: Generazione con documenti...")
        
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
            print("✅ GeneratorAgent: Risposta generata con documenti")
        except Exception as e:
            state["final_answer"] = f"Errore nella generazione: {str(e)}"
            state["confidence"] = 0.0
            print(f"❌ GeneratorAgent: Errore - {str(e)}")
        
        return state
    
    def generate_with_data(self, state: GraphState) -> GraphState:
        """Genera risposta usando i dati del dataset"""
        print("💬 GeneratorAgent: Generazione con dati...")
        
        question = state["question"]
        data = state["retrieved_data"]
        
        # Formatta i dati in modo leggibile
        data_context = self._format_data(data)
        
        messages = [
            {
                "role": "system",
                "content": """Sei un assistente HR che analizza dati aziendali.
Presenta i dati in modo chiaro e professionale.
Usa formattazione, numeri e bullet points quando appropriato.
Usa sempre l'italiano."""
            },
            {
                "role": "user",
                "content": f"""DATI RECUPERATI:
{data_context}

DOMANDA: {question}

Analizza i dati e fornisci una risposta chiara."""
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
            state["confidence"] = 0.85
            print("✅ GeneratorAgent: Risposta generata con dati")
        except Exception as e:
            state["final_answer"] = f"Errore nella generazione: {str(e)}"
            state["confidence"] = 0.0
            print(f"❌ GeneratorAgent: Errore - {str(e)}")
        
        return state
    
    def generate_hybrid(self, state: GraphState) -> GraphState:
        """Genera risposta combinando documenti e dati"""
        print("💬 GeneratorAgent: Generazione ibrida (documenti + dati)...")
        
        question = state["question"]
        documents = state["retrieved_documents"]
        data = state["retrieved_data"]
        
        doc_context = "\n\n".join([f"[Doc {i+1}] {doc}" for i, doc in enumerate(documents)])
        data_context = self._format_data(data)
        
        messages = [
            {
                "role": "system",
                "content": """Sei un assistente HR esperto di TechCorp Solutions.
Combina informazioni da policy aziendali e dati reali dei dipendenti.
Fornisci risposte complete che integrano normative e statistiche.
Usa sempre l'italiano."""
            },
            {
                "role": "user",
                "content": f"""POLICY E DOCUMENTI:
{doc_context}

DATI DIPENDENTI:
{data_context}

DOMANDA: {question}

Fornisci una risposta completa che integra policy e dati."""
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
            state["confidence"] = 0.95
            print("✅ GeneratorAgent: Risposta ibrida generata")
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
            state["data_used"] = False
            print("✅ GeneratorAgent: Risposta diretta generata")
        except Exception as e:
            state["final_answer"] = f"Errore nella generazione: {str(e)}"
            state["confidence"] = 0.0
            print(f"❌ GeneratorAgent: Errore - {str(e)}")
        
        return state
    
    def _format_data(self, data: List[dict]) -> str:
        """Formatta i dati per il prompt"""
        formatted = []
        for item in data:
            item_type = item.get('type', 'unknown')
            formatted.append(f"[{item_type.upper()}]")
            formatted.append(str(item))
            formatted.append("")
        return "\n".join(formatted)