import os
from dotenv import load_dotenv
from graph import AdaptiveRAGGraph


def run_examples(rag_system: AdaptiveRAGGraph):
    """Esegue esempi che dimostrano il routing adattivo a 4 vie"""
    
    print("\n" + "="*70)
    print("🧪 TEST 1: Domande DIRECT (risposta generica)")
    print("="*70)
    
    direct_questions = [
        "Ciao! Come stai?",
        "Grazie per l'aiuto!",
        "Puoi spiegarmi come funzioni?",
    ]
    
    for q in direct_questions:
        rag_system.query(q)
    
    print("\n" + "="*70)
    print("🧪 TEST 2: Domande DOCUMENTS (policy e procedure)")
    print("="*70)
    
    document_questions = [
        "Come funziona la policy sulle ferie?",
        "Qual è il preavviso per le dimissioni?",
        "Quali benefit offre l'azienda?",
        "Come richiedere i rimborsi spese?",
    ]
    
    for q in document_questions:
        rag_system.query(q)
    
    print("\n" + "="*70)
    print("🧪 TEST 3: Domande DATASET (dati strutturati)")
    print("="*70)
    
    dataset_questions = [
        "Quanti dipendenti abbiamo in totale?",
        "Chi ha usato più ferie quest'anno?",
        "Qual è lo stipendio medio per dipartimento?",
        "Quanti giorni di smart working in media?",
        "Chi sono i top performer?",
    ]
    
    for q in dataset_questions:
        rag_system.query(q)
    
    print("\n" + "="*70)
    print("🧪 TEST 4: Domande HYBRID (policy + dati)")
    print("="*70)
    
    hybrid_questions = [
        "Mostrami chi ha usato più ferie e spiegami la policy aziendale sulle ferie",
        "Dammi un'analisi completa sullo smart working: policy e dati dei dipendenti",
        "Confronta i dati di performance con le policy di sviluppo carriera",
    ]
    
    for q in hybrid_questions:
        rag_system.query(q)


def interactive_mode(rag_system: AdaptiveRAGGraph):
    """Modalità interattiva"""
    print("\n" + "="*70)
    print("💬 MODALITÀ INTERATTIVA")
    print("="*70)
    print("Fai domande al sistema. Comandi speciali:")
    print("  • 'exit', 'quit', 'esci' → Termina")
    print("  • 'graph' → Visualizza struttura sistema")
    print("  • 'help' → Mostra esempi di domande\n")
    
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
            
            if question.lower() == 'help':
                print("\n📚 Esempi di domande:")
                print("\nDIRECT:")
                print("  • Ciao, come va?")
                print("\nDOCUMENTS:")
                print("  • Come funziona la policy ferie?")
                print("  • Qual è il dress code aziendale?")
                print("\nDATASET:")
                print("  • Quanti dipendenti abbiamo?")
                print("  • Chi lavora a Milano?")
                print("\nHYBRID:")
                print("  • Analisi completa smart working con policy e dati")
                print("  • Chi ha più ferie e qual è la policy?\n")
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
    dataset_path = "hr_dataset.csv"
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ Errore: {knowledge_base_path} non trovato")
        return
    
    if not os.path.exists(dataset_path):
        print(f"❌ Errore: {dataset_path} non trovato")
        return
    
    # Inizializza il sistema
    rag_system = AdaptiveRAGGraph(hf_token, knowledge_base_path, dataset_path)
    
    # Mostra struttura
    rag_system.visualize_graph()
    
    # Esegui esempi dimostrativi
    run_examples(rag_system)
    
    # Modalità interattiva
    interactive_mode(rag_system)


if __name__ == "__main__":
    main()