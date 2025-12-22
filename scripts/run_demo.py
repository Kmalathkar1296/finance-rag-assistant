import sys
sys.path.append('.')

from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem
from src.config import Config
from src.utils import print_section, print_subsection

def main():
    Config.ensure_directories()
    
    print_section("FINANCE RAG ASSISTANT - DEMO")
    
    # Generate data
    print_subsection("Step 1: Generating Sample Data")
    generator = FinanceDataGenerator()
    ar_df = generator.generate_accounts_receivable(n=50)
    payments_df = generator.generate_payments(ar_df)
    gl_df = generator.generate_general_ledger(ar_df)
    budget_df = generator.generate_budget_forecast(n_years=1)
    claims_df = generator.generate_expense_claims(n=100)
    print("✓ Sample data generated")
    
    # Initialize RAG
    print_subsection("Step 2: Initializing RAG System")
    rag = FinanceRAGSystem()
    rag.load_data(ar_df, payments_df, gl_df, budget_df, claims_df)
    rag.build_vector_store()
    print("✓ RAG system ready")
    
    # Run queries
    print_subsection("Step 3: Running Sample Queries")
    
    queries = [
        "Show me all payment discrepancies",
        "Which payments are overdue?",
        "Which departments are over budget?",
        "Show me pending expense claims"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 70)
        result = rag.query(query)
        print(f"   {result.get('summary', 'No summary available')}")
    
    # Generate report
    print_subsection("Step 4: Generating Report")
    report_path = f"outputs/reports/demo_report_{Config.get_timestamp()}.txt"
    rag.generate_report(report_path)
    print(f"✓ Report saved to: {report_path}")
    
    print_section("DEMO COMPLETE")
    print("\nNext steps:")
    print("1. Try interactive mode: python scripts/interactive_query.py")
    print("2. Load your own data from Excel files")
    print("3. Explore the generated report")

if __name__ == "__main__":
    main()