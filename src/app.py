"""
Finance RAG Assistant - Gradio Web Application
Deployable to HuggingFace Spaces
"""

import gradio as gr
import pandas as pd
import os
from datetime import datetime

from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem

# ============================================================================
# GLOBAL STATE
# ============================================================================

class AppState:
    def __init__(self):
        self.generator = FinanceDataGenerator()
        self.rag_system = None
        self.ar_df = None
        self.payments_df = None
        self.gl_df = None
        self.budget_df = None
        self.claims_df = None
        self.is_initialized = False

state = AppState()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def setup_system(n_invoices, n_claims, progress=gr.Progress()):
    """Initialize RAG system with sample data"""
    global state
    
    try:
        progress(0, desc="Generating invoices...")
        state.ar_df = state.generator.generate_accounts_receivable(n=n_invoices)
        
        progress(0.2, desc="Generating payments...")
        state.payments_df = state.generator.generate_payments(state.ar_df)
        
        progress(0.4, desc="Generating general ledger...")
        state.gl_df = state.generator.generate_general_ledger(state.ar_df)
        
        progress(0.6, desc="Generating budget data...")
        state.budget_df = state.generator.generate_budget_forecast(n_years=1)
        
        progress(0.8, desc="Generating expense claims...")
        state.claims_df = state.generator.generate_expense_claims(n=n_claims)
        
        progress(0.9, desc="Building RAG system...")
        state.rag_system = FinanceRAGSystem()
        state.rag_system.load_data(
            state.ar_df,
            state.payments_df,
            state.gl_df,
            state.budget_df,
            state.claims_df
        )
        state.rag_system.build_vector_store()
        state.is_initialized = True
        
        progress(1.0, desc="Complete!")
        
        return (
            f"✅ System initialized successfully!\n\n"
            f"📊 Generated:\n"
            f"  • {len(state.ar_df)} invoices\n"
            f"  • {len(state.payments_df)} payments\n"
            f"  • {len(state.budget_df)} budget records\n"
            f"  • {len(state.claims_df)} expense claims\n\n"
            f"🔍 You can now ask questions!"
        )
    except Exception as e:
        return f"❌ Error: {str(e)}"

def query_system(question):
    """Query the RAG system"""
    if not state.is_initialized:
        return "⚠️ Please initialize the system first using the Setup tab!"
    
    try:
        result = state.rag_system.query(question)
        
        # Format response
        response = f"### 📊 Query Results\n\n"
        response += f"**Question:** {question}\n\n"
        response += f"**Summary:** {result.get('summary', 'No summary available')}\n\n"
        
        if 'analysis' in result:
            response += f"**Analysis:**\n{result['analysis']}\n\n"
        
        if 'recommendations' in result and result['recommendations']:
            response += f"**Recommendations:**\n"
            for rec in result['recommendations']:
                response += f"- {rec}\n"
        
        return response
    
    except Exception as e:
        return f"❌ Error processing query: {str(e)}"

def get_discrepancies():
    """Get all discrepancies"""
    if not state.is_initialized:
        return "⚠️ Please initialize the system first!"
    
    try:
        discrepancies = state.rag_system.find_discrepancies()
        
        if not discrepancies:
            return "✅ No discrepancies found! All payments match invoices."
        
        # Convert to DataFrame for display
        df = pd.DataFrame(discrepancies)
        
        summary = f"### ⚠️ Found {len(discrepancies)} Discrepancies\n\n"
        summary += f"**Critical:** {sum(1 for d in discrepancies if d['severity'] == 'CRITICAL')}\n"
        summary += f"**High:** {sum(1 for d in discrepancies if d['severity'] == 'HIGH')}\n"
        summary += f"**Medium:** {sum(1 for d in discrepancies if d['severity'] == 'MEDIUM')}\n\n"
        
        return summary, df
    
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame()

def get_data_overview():
    """Get data overview"""
    if not state.is_initialized:
        return "⚠️ Please initialize the system first!", pd.DataFrame(), pd.DataFrame()
    
    try:
        # Summary stats
        total_invoices = len(state.ar_df)
        total_payments = len(state.payments_df)
        outstanding = state.ar_df[state.ar_df['Status'] != 'Paid']['Amount'].sum()
        
        summary = f"### 📊 Financial Overview\n\n"
        summary += f"**Total Invoices:** {total_invoices}\n"
        summary += f"**Total Payments:** {total_payments}\n"
        summary += f"**Outstanding Amount:** ${outstanding:,.2f}\n"
        
        return summary, state.ar_df.head(10), state.payments_df.head(10)
    
    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame(), pd.DataFrame()

def generate_report():
    """Generate comprehensive report"""
    if not state.is_initialized:
        return "⚠️ Please initialize the system first!"
    
    try:
        import tempfile
        
        # Generate report
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w')
        state.rag_system.generate_report(temp_file.name)
        
        # Read content
        with open(temp_file.name, 'r') as f:
            report_content = f.read()
        
        # Clean up
        os.unlink(temp_file.name)
        
        return report_content
    
    except Exception as e:
        return f"❌ Error generating report: {str(e)}"

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main-header {
    text-align: center;
    color: #2563eb;
    font-size: 2.5rem;
    font-weight: bold;
    margin-bottom: 1rem;
}
.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.2rem;
    margin-bottom: 2rem;
}
"""

# Create Gradio Blocks interface
with gr.Blocks(css=custom_css, title="Finance RAG Assistant") as demo:
    
    # Header
    gr.Markdown("""
    <div class="main-header">💰 Finance RAG Assistant</div>
    <div class="subtitle">AI-Powered Payment Reconciliation & Financial Analysis</div>
    """)
    
    # Tabs
    with gr.Tabs():
        
        # ========================================================================
        # TAB 1: SETUP
        # ========================================================================
        with gr.Tab("🚀 Setup"):
            gr.Markdown("""
            ### Initialize the System
            Generate sample financial data and build the RAG system.
            """)
            
            with gr.Row():
                with gr.Column():
                    n_invoices = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Number of Invoices"
                    )
                    n_claims = gr.Slider(
                        minimum=20,
                        maximum=400,
                        value=100,
                        step=20,
                        label="Number of Expense Claims"
                    )
                    setup_btn = gr.Button("🎲 Generate Data & Initialize", variant="primary", size="lg")
                
                with gr.Column():
                    setup_output = gr.Textbox(
                        label="Status",
                        lines=12,
                        placeholder="Click the button to initialize..."
                    )
            
            setup_btn.click(
                fn=setup_system,
                inputs=[n_invoices, n_claims],
                outputs=setup_output
            )
        
        # ========================================================================
        # TAB 2: QUERY ASSISTANT
        # ========================================================================
        with gr.Tab("🔍 Query Assistant"):
            gr.Markdown("""
            ### Ask Questions About Your Financial Data
            Use natural language to query invoices, payments, budgets, and expenses.
            """)
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Show me all payment discrepancies",
                    lines=2
                )
            
            with gr.Row():
                query_btn = gr.Button("🔍 Search", variant="primary")
                clear_btn = gr.Button("🗑️ Clear")
            
            # Sample queries
            gr.Markdown("#### Quick Queries:")
            with gr.Row():
                sample_q1 = gr.Button("💸 Payment Discrepancies", size="sm")
                sample_q2 = gr.Button("⏰ Overdue Payments", size="sm")
                sample_q3 = gr.Button("📊 Budget Variance", size="sm")
                sample_q4 = gr.Button("💳 Pending Claims", size="sm")
            
            query_output = gr.Markdown(label="Results")
            
            # Connect buttons
            query_btn.click(fn=query_system, inputs=query_input, outputs=query_output)
            clear_btn.click(lambda: ("", ""), outputs=[query_input, query_output])
            
            sample_q1.click(lambda: "Show me all payment discrepancies", outputs=query_input)
            sample_q2.click(lambda: "Which payments are overdue?", outputs=query_input)
            sample_q3.click(lambda: "Which departments are over budget?", outputs=query_input)
            sample_q4.click(lambda: "Show me pending expense claims", outputs=query_input)
        
        # ========================================================================
        # TAB 3: DISCREPANCIES
        # ========================================================================
        with gr.Tab("⚠️ Discrepancies"):
            gr.Markdown("""
            ### Payment Discrepancies & Issues
            Automatically detect and analyze payment mismatches and overdue invoices.
            """)
            
            disc_btn = gr.Button("🔍 Analyze Discrepancies", variant="primary")
            
            disc_summary = gr.Markdown(label="Summary")
            disc_table = gr.Dataframe(
                label="Discrepancy Details",
                wrap=True,
                interactive=False
            )
            
            disc_btn.click(
                fn=get_discrepancies,
                outputs=[disc_summary, disc_table]
            )
        
        # ========================================================================
        # TAB 4: DATA OVERVIEW
        # ========================================================================
        with gr.Tab("📊 Data Overview"):
            gr.Markdown("""
            ### Financial Data Overview
            View key metrics and browse your financial data.
            """)
            
            overview_btn = gr.Button("📊 Refresh Data", variant="primary")
            
            overview_summary = gr.Markdown(label="Summary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Recent Invoices")
                    ar_table = gr.Dataframe(label="Accounts Receivable")
                
                with gr.Column():
                    gr.Markdown("#### Recent Payments")
                    payments_table = gr.Dataframe(label="Payments")
            
            overview_btn.click(
                fn=get_data_overview,
                outputs=[overview_summary, ar_table, payments_table]
            )
        
        # ========================================================================
        # TAB 5: REPORTS
        # ========================================================================
        with gr.Tab("📄 Reports"):
            gr.Markdown("""
            ### Generate Comprehensive Reports
            Create detailed financial analysis reports.
            """)
            
            report_btn = gr.Button("📊 Generate Report", variant="primary")
            report_output = gr.Textbox(
                label="Report",
                lines=25,
                max_lines=50
            )
            
            report_btn.click(fn=generate_report, outputs=report_output)
    
    # Footer
    gr.Markdown("""
    ---
    <div style='text-align: center; color: #64748b; font-size: 0.9rem;'>
    Built with ❤️ using <strong>LangChain</strong>, <strong>ChromaDB</strong>, and <strong>Gradio</strong><br>
    Finance RAG Assistant v1.0.0
    </div>
    """)

# Launch
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )