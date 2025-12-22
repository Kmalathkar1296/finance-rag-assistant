import gradio as gr
import os
from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem

# Initialize
generator = FinanceDataGenerator()
rag = None

def setup_system(n_records):
    """Setup RAG system with sample data"""
    global rag
    
    ar_df = generator.generate_accounts_receivable(n=n_records)
    payments_df = generator.generate_payments(ar_df)
    gl_df = generator.generate_general_ledger(ar_df)
    budget_df = generator.generate_budget_forecast()
    claims_df = generator.generate_expense_claims(n=n_records*2)
    
    rag = FinanceRAGSystem()
    rag.load_data(ar_df, payments_df, gl_df, budget_df, claims_df)
    rag.build_vector_store()
    
    return f"✅ System ready with {n_records} invoices!"

def query_system(question):
    """Query the RAG system"""
    if rag is None:
        return "Please setup the system first!"
    
    result = rag.query(question)
    return result.get('summary', 'No results found')

# Create Gradio interface
with gr.Blocks(title="Finance RAG Assistant") as demo:
    gr.Markdown("# 💰 Finance RAG Assistant")
    gr.Markdown("AI-powered payment reconciliation and financial analysis")
    
    with gr.Row():
        with gr.Column():
            n_records = gr.Slider(10, 100, value=50, label="Number of Invoices")
            setup_btn = gr.Button("🚀 Setup System", variant="primary")
            setup_output = gr.Textbox(label="Status")
        
        with gr.Column():
            query_input = gr.Textbox(label="Ask a Question", 
                                     placeholder="e.g., Show me all discrepancies")
            query_btn = gr.Button("🔍 Search")
            query_output = gr.Textbox(label="Results", lines=10)
    
    # Sample queries
    gr.Examples(
        examples=[
            "Show me all payment discrepancies",
            "Which payments are overdue?",
            "Which departments are over budget?",
            "Show me pending expense claims"
        ],
        inputs=query_input
    )
    
    setup_btn.click(setup_system, inputs=n_records, outputs=setup_output)
    query_btn.click(query_system, inputs=query_input, outputs=query_output)

demo.launch()