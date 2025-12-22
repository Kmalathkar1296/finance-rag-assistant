# Finance RAG Assistant

AI-powered payment reconciliation system using LangChain, ChromaDB, and open-source embeddings.

## Features

- 🔍 Automated payment discrepancy detection
- 📊 Budget variance analysis
- 💳 Expense claims management
- 🤖 RAG-based natural language queries
- 📈 Comprehensive financial reporting

## Quick Start


# Clone the repository
git clone <your-repo-url>
cd finance-rag-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run demo
python scripts/run_demo.py

## Usage

### Generate Sample Data

python scripts/generate_data.py --records 100


### Interactive Query Mode

python scripts/interactive_query.py

### Load Your Own Data

from src.rag_system import FinanceRAGSystem
import pandas as pd

# Load your Excel files
ar_df = pd.read_excel('data/raw/accounts_receivable.xlsx')
payments_df = pd.read_excel('data/raw/payments.xlsx')
gl_df = pd.read_excel('data/raw/general_ledger.xlsx')
budget_df = pd.read_excel('data/raw/budget_forecast.xlsx')
claims_df = pd.read_excel('data/raw/expense_claims.xlsx')

# Initialize RAG system
rag = FinanceRAGSystem()
rag.load_data(ar_df, payments_df, gl_df, budget_df, claims_df)
rag.build_vector_store()

# Query
result = rag.query("Show me all payment discrepancies")
print(result['summary'])


## Sample Queries

- "Show me all discrepancies in payments"
- "Which payments are overdue?"
- "Which departments are over budget?"
- "Show me pending expense claims"
- "Which expense claims exceed policy limits?"

## Data Sources

Download real datasets from [Excelx.com](https://excelx.com/practice-data/finance-accounting/)
