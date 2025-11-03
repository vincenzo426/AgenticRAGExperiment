import os
from dotenv import load_dotenv
from graph import AdaptiveRAGGraph


def run_examples(rag_system: AdaptiveRAGGraph):
    """Esegue esempi che dimostrano il routing adattivo"""
    
    print("\n" + "="*70)
    print("🧪 TEST: Domande che NON richiedono RAG (risposta diretta)")
    print("="*70)
    
    direct_questions = [
        "Ciao! Come stai?",
        "Puoi aiutarmi?",
        "Grazie mille per l'aiuto!",
    ]
    
    for q in direct_questions:
        rag_system.query(q)
    
    print("\n" + "="*70)
    print("🧪 TEST: Domande che richiedono RAG (informazioni specifiche)")
    print("="*70)
    
    rag_questions = [
        "Quanti giorni di ferie ho all'anno?",
        "Come funziona lo smart working in azienda?",
        "Quali benefit offre TechCorp?",
        "Qual è il preavviso per le dimissioni?",
        "Come vengono rimborsate le spese?",
        "Che formazione ricevo in onboarding?",
    ]
    
    for q in rag_questions:
        rag_system.query(q)


def interactive_mode(rag_system: AdaptiveRAGGraph):
    """Modalità interattiva"""
    print("\n" + "="*70)
    print("💬 MODALITÀ INTERATTIVA")
    print("="*70)
    print("Fai domande al sistema. Digita 'exit', 'quit' o 'esci' per terminare.")
    print("Digita 'graph' per visualizzare la struttura del sistema.\n")
    
    while True:
        try:
            question = input("Tu: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['exit', 'quit', 'esci']:
                print("\n👋 Arrivederci!")
                break
            
            if question.lower() == 'graph':
                rag_system.visualize_graph()
                continue
            
            rag_system.query(question)
            
        except KeyboardInterrupt:
            print("\n👋 Arrivederci!")
            break
        except Exception as e:
            print(f"❌ Errore: {e}")


def main():
    # Carica variabili d'ambiente
    load_dotenv()
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("❌ Errore: HF_TOKEN non trovato nel file .env")
        print("\nCrea un file .env con:")
        print("HF_TOKEN=your_huggingface_token")
        return
    
    knowledge_base_path = "knowledge_base.txt"
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ Errore: {knowledge_base_path} non trovato")
        return
    
    # Inizializza il sistema
    rag_system = AdaptiveRAGGraph(hf_token, knowledge_base_path)
    
    # Mostra struttura
    rag_system.visualize_graph()
    
    # Esegui esempi dimostrativi
    run_examples(rag_system)
    
    # Modalità interattiva
    interactive_mode(rag_system)


if __name__ == "__main__":
    main()
