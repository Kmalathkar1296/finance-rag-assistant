import sys
sys.path.append('.')

from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem
from src.config import Config
from src.utils import print_section
import os

def main():
    Config.ensure_directories()
    
    print_section("FINANCE RAG ASSISTANT - Interactive Mode")
    
    # Check if data exists
    sample_data_path = "data/sample/accounts_receivable.xlsx"
    
    if not os.path.exists(sample_data_path):
        print("No sample data found. Generating...")
        generator = FinanceDataGenerator()
        ar_df = generator.generate_accounts_receivable(n=100)
        payments_df = generator.generate_payments(ar_df)
        gl_df = generator.generate_general_ledger(ar_df)
        budget_df = generator.generate_budget_forecast()
        claims_df = generator.generate_expense_claims()
    else:
        print("Loading existing data...")
        import pandas as pd
        ar_df = pd.read_excel("data/sample/accounts_receivable.xlsx")
        payments_df = pd.read_excel("data/sample/payments.xlsx")
        gl_df = pd.read_excel("data/sample/general_ledger.xlsx")
        budget_df = pd.read_excel("data/sample/budget_forecast.xlsx")
        claims_df = pd.read_excel("data/sample/expense_claims.xlsx")
    
    # Initialize RAG
    print("\nInitializing RAG system...")
    rag = FinanceRAGSystem()
    rag.load_data(ar_df, payments_df, gl_df, budget_df, claims_df)
    rag.build_vector_store()
    
    print("\n" + "=" * 80)
    print("RAG System Ready! Ask questions about your financial data.")
    print("=" * 80)
    print("\nExample queries:")
    print("  - Show me all payment discrepancies")
    print("  - Which customers have overdue invoices?")
    print("  - What's the budget variance for Marketing?")
    print("  - Show me expense claims over policy limits")
    print("\nType 'exit' or 'quit' to end the session.")
    print("=" * 80 + "\n")
    
    while True:
        try:
            query = input("\n💬 Your query: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if not query:
                continue
            
            print("\n🔍 Analyzing...")
            result = rag.query(query)
            
            print("\n📊 Result:")
            print("-" * 80)
            print(result.get('summary', 'No results found'))
            
            if 'analysis' in result:
                print(result['analysis'])
            
            if 'recommendations' in result:
                print("\n💡 Recommendations:")
                for rec in result['recommendations']:
                    print(f"  • {rec}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()