"""
Finance RAG Assistant - Gradio Web Application
Deployable to HuggingFace Spaces with full feature parity to Streamlit version
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem

# ============================================================================
# GLOBAL STATE MANAGEMENT
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.generator = FinanceDataGenerator()
        self.rag_system = None
        self.ar_df = None
        self.payments_df = None
        self.gl_df = None
        self.budget_df = None
        self.claims_df = None
        self.is_initialized = False
        self.last_query = None

state = AppState()

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def setup_system(n_invoices, n_claims, progress=gr.Progress()):
    """Initialize RAG system with sample data"""
    global state
    
    try:
        progress(0, desc="🎲 Generating financial data...")
        
        # Generate data with progress updates
        progress(0.1, desc="📄 Creating invoices...")
        state.ar_df = state.generator.generate_accounts_receivable(n=n_invoices)
        
        progress(0.3, desc="💰 Creating payments...")
        state.payments_df = state.generator.generate_payments(state.ar_df)
        
        progress(0.4, desc="📊 Creating ledger entries...")
        state.gl_df = state.generator.generate_general_ledger(state.ar_df)
        
        progress(0.6, desc="💼 Creating budget data...")
        state.budget_df = state.generator.generate_budget_forecast(n_years=1)
        
        progress(0.8, desc="💳 Creating expense claims...")
        state.claims_df = state.generator.generate_expense_claims(n=n_claims)
        
        progress(0.9, desc="🔧 Building RAG system...")
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
        
        progress(1.0, desc="✅ Complete!")
        
        # Summary statistics
        total_outstanding = state.ar_df[state.ar_df['Status'] != 'Paid']['Amount'].sum()
        total_collected = state.ar_df[state.ar_df['Status'] == 'Paid']['Amount'].sum()
        
        summary = f"""
## ✅ System Initialized Successfully!

### 📊 Data Generated:
- **Invoices:** {len(state.ar_df)} records
- **Payments:** {len(state.payments_df)} records
- **Budget Records:** {len(state.budget_df)} entries
- **Expense Claims:** {len(state.claims_df)} claims

### 💰 Financial Summary:
- **Total Outstanding:** ${total_outstanding:,.2f}
- **Total Collected:** ${total_collected:,.2f}
- **Collection Rate:** {(total_collected/(total_outstanding+total_collected)*100):.1f}%

🔍 **You can now ask questions about your financial data!**
        """
        
        return summary
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}\n\nPlease try again with different parameters."


def query_system(question):
    """Query the RAG system"""
    if not state.is_initialized:
        return "⚠️ **Please initialize the system first!**\n\nGo to the **Setup** tab and click 'Generate Data'."
    
    if not question or not question.strip():
        return "⚠️ **Please enter a question.**"
    
    try:
        result = state.rag_system.query(question)
        state.last_query = question
        
        # Format response with markdown
        response = f"## 📊 Query Results\n\n"
        response += f"**Question:** {question}\n\n"
        response += f"### Summary\n{result.get('summary', 'No summary available')}\n\n"
        
        if 'analysis' in result:
            response += f"### 📈 Detailed Analysis\n```\n{result['analysis']}\n```\n\n"
        
        if 'recommendations' in result and result['recommendations']:
            response += f"### 💡 Recommendations\n"
            for i, rec in enumerate(result['recommendations'], 1):
                response += f"{i}. {rec}\n"
            response += "\n"
        
        return response
        
    except Exception as e:
        return f"❌ **Error processing query:** {str(e)}"


def get_discrepancies():
    """Get all discrepancies"""
    if not state.is_initialized:
        return "⚠️ **Please initialize the system first!**", pd.DataFrame()
    
    try:
        discrepancies = state.rag_system.find_discrepancies()
        
        if not discrepancies:
            return "✅ **No discrepancies found!** All payments match invoices perfectly.", pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(discrepancies)
        
        # Summary with statistics
        critical = sum(1 for d in discrepancies if d['severity'] == 'CRITICAL')
        high = sum(1 for d in discrepancies if d['severity'] == 'HIGH')
        medium = sum(1 for d in discrepancies if d['severity'] == 'MEDIUM')
        total_variance = sum(d['difference'] for d in discrepancies)
        
        summary = f"""
## ⚠️ Discrepancies Found: {len(discrepancies)}

### Severity Breakdown:
- 🔴 **Critical:** {critical} issues
- 🟠 **High:** {high} issues
- 🟡 **Medium:** {medium} issues

### Financial Impact:
- **Total Variance:** ${abs(total_variance):,.2f}

### Recommended Actions:
1. Review all critical discrepancies immediately
2. Contact customers with overdue payments
3. Verify payment processing systems
4. Update ledger entries as needed
        """
        
        return summary, df
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}", pd.DataFrame()


def get_data_overview():
    """Get data overview with statistics"""
    if not state.is_initialized:
        return "⚠️ **Please initialize the system first!**", pd.DataFrame(), pd.DataFrame()
    
    try:
        # Calculate statistics
        total_invoices = len(state.ar_df)
        total_payments = len(state.payments_df)
        outstanding = state.ar_df[state.ar_df['Status'] != 'Paid']['Amount'].sum()
        collected = state.ar_df[state.ar_df['Status'] == 'Paid']['Amount'].sum()
        overdue = len(state.ar_df[(state.ar_df['Status'] != 'Paid') & 
                                   (pd.to_datetime(state.ar_df['DueDate']) < datetime.now())])
        
        summary = f"""
## 📊 Financial Data Overview

### Key Metrics:
- **Total Invoices:** {total_invoices}
- **Total Payments:** {total_payments}
- **Outstanding Amount:** ${outstanding:,.2f}
- **Collected Amount:** ${collected:,.2f}
- **Overdue Invoices:** {overdue}

### Collection Performance:
- **Collection Rate:** {(collected/(collected+outstanding)*100):.1f}%
- **Outstanding Rate:** {(outstanding/(collected+outstanding)*100):.1f}%
        """
        
        return summary, state.ar_df.head(15), state.payments_df.head(15)
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}", pd.DataFrame(), pd.DataFrame()


def create_status_chart():
    """Create invoice status pie chart"""
    if not state.is_initialized:
        return None
    
    try:
        status_counts = state.ar_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Invoice Status Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.3
        )
        fig.update_layout(height=400)
        return fig
    except:
        return None


def create_customer_chart():
    """Create customer amount bar chart"""
    if not state.is_initialized:
        return None
    
    try:
        customer_amounts = state.ar_df.groupby('Customer')['Amount'].sum().sort_values(ascending=False).head(8)
        fig = px.bar(
            x=customer_amounts.index,
            y=customer_amounts.values,
            title="Top 8 Customers by Invoice Amount",
            labels={'x': 'Customer', 'y': 'Amount ($)'},
            color=customer_amounts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    except:
        return None


def create_budget_chart():
    """Create budget vs actual chart"""
    if not state.is_initialized or state.budget_df is None:
        return None
    
    try:
        budget_by_dept = state.budget_df.groupby('Dept').agg({
            'BudgetUSD': 'sum',
            'ActualUSD': 'sum'
        }).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Budget',
            x=budget_by_dept['Dept'],
            y=budget_by_dept['BudgetUSD'],
            marker_color='lightblue'
        ))
        fig.add_trace(go.Bar(
            name='Actual',
            x=budget_by_dept['Dept'],
            y=budget_by_dept['ActualUSD'],
            marker_color='darkblue'
        ))
        fig.update_layout(
            barmode='group',
            title="Budget vs Actual Spending by Department",
            height=400,
            xaxis_title="Department",
            yaxis_title="Amount ($)"
        )
        return fig
    except:
        return None


def create_expense_category_chart():
    """Create expense by category chart"""
    if not state.is_initialized or state.claims_df is None:
        return None
    
    try:
        category_amounts = state.claims_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
        fig = px.bar(
            x=category_amounts.values,
            y=category_amounts.index,
            orientation='h',
            title="Expense Claims by Category",
            labels={'x': 'Amount ($)', 'y': 'Category'},
            color=category_amounts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        return fig
    except:
        return None


def create_expense_status_chart():
    """Create expense claims status chart"""
    if not state.is_initialized or state.claims_df is None:
        return None
    
    try:
        status_counts = state.claims_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Expense Claim Status",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.3
        )
        fig.update_layout(height=400)
        return fig
    except:
        return None


def generate_report():
    """Generate comprehensive report"""
    if not state.is_initialized:
        return "⚠️ **Please initialize the system first!**"
    
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
        return f"❌ **Error generating report:** {str(e)}"


def refresh_analytics():
    """Refresh all analytics charts"""
    return (
        create_status_chart(),
        create_customer_chart(),
        create_budget_chart(),
        create_expense_category_chart(),
        create_expense_status_chart()
    )

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    max-width: 1400px !important;
}

.main-header {
    text-align: center;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

.metric-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 12px;
    color: white;
    text-align: center;
}

.quick-action-btn {
    background: #f0f4f8;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem;
    cursor: pointer;
    transition: all 0.3s;
}

.quick-action-btn:hover {
    background: #e2e8f0;
    transform: translateY(-2px);
}

footer {
    text-align: center;
    padding: 2rem;
    color: #94a3b8;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Finance RAG Assistant", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML("""
        <div class="main-header">💰 Finance RAG Assistant</div>
        <div class="subtitle">AI-Powered Payment Reconciliation & Financial Analysis</div>
    """)
    
    # Main tabs
    with gr.Tabs() as tabs:
        
        # ====================================================================
        # TAB 1: SETUP
        # ====================================================================
        with gr.Tab("🚀 Setup", id="setup"):
            gr.Markdown("""
            ## Initialize Your Financial Analysis System
            
            Generate sample financial data and build the AI-powered RAG system for analysis.
            This includes invoices, payments, budget data, and expense claims.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuration")
                    
                    n_invoices = gr.Slider(
                        minimum=10,
                        maximum=200,
                        value=50,
                        step=10,
                        label="Number of Invoices",
                        info="More invoices = more comprehensive analysis"
                    )
                    
                    n_claims = gr.Slider(
                        minimum=20,
                        maximum=400,
                        value=100,
                        step=20,
                        label="Number of Expense Claims",
                        info="Typically 2x the number of invoices"
                    )
                    
                    setup_btn = gr.Button(
                        "🎲 Generate Data & Initialize System",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Status")
                    setup_output = gr.Markdown(
                        value="Click the button to generate sample data and initialize the system...",
                        label="System Status"
                    )
            
            setup_btn.click(
                fn=setup_system,
                inputs=[n_invoices, n_claims],
                outputs=setup_output
            )
        
        # ====================================================================
        # TAB 2: QUERY ASSISTANT
        # ====================================================================
        with gr.Tab("🔍 Query Assistant", id="query"):
            gr.Markdown("""
            ## Ask Questions About Your Financial Data
            
            Use natural language to query invoices, payments, budgets, and expense claims.
            The AI will search through your data and provide detailed insights.
            """)
            
            with gr.Row():
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., Show me all payment discrepancies",
                    lines=3,
                    scale=4
                )
            
            with gr.Row():
                query_btn = gr.Button("🔍 Search", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ Clear", scale=1)
            
            gr.Markdown("### 💡 Quick Queries")
            with gr.Row():
                sample_q1 = gr.Button("💸 Payment Discrepancies", size="sm")
                sample_q2 = gr.Button("⏰ Overdue Payments", size="sm")
                sample_q3 = gr.Button("📊 Budget Variance", size="sm")
                sample_q4 = gr.Button("💳 Pending Claims", size="sm")
            
            query_output = gr.Markdown(
                label="Results",
                value="Enter a question above and click Search to get started..."
            )
            
            # Connect buttons
            query_btn.click(fn=query_system, inputs=query_input, outputs=query_output)
            clear_btn.click(lambda: ("", ""), outputs=[query_input, query_output])
            
            sample_q1.click(lambda: "Show me all payment discrepancies", outputs=query_input)
            sample_q2.click(lambda: "Which payments are overdue?", outputs=query_input)
            sample_q3.click(lambda: "Which departments are over budget?", outputs=query_input)
            sample_q4.click(lambda: "Show me pending expense claims", outputs=query_input)
        
        # ====================================================================
        # TAB 3: DISCREPANCIES
        # ====================================================================
        with gr.Tab("⚠️ Discrepancies", id="discrepancies"):
            gr.Markdown("""
            ## Payment Discrepancies & Issues
            
            Automatically detect and analyze payment mismatches, overdue invoices, and missing records.
            """)
            
            disc_btn = gr.Button("🔍 Analyze Discrepancies", variant="primary", size="lg")
            
            disc_summary = gr.Markdown(label="Summary")
            disc_table = gr.Dataframe(
                label="Discrepancy Details",
                wrap=True,
                interactive=False,
                height=400
            )
            
            disc_btn.click(
                fn=get_discrepancies,
                outputs=[disc_summary, disc_table]
            )
        
        # ====================================================================
        # TAB 4: DATA OVERVIEW
        # ====================================================================
        with gr.Tab("📊 Data Overview", id="overview"):
            gr.Markdown("""
            ## Financial Data Overview
            
            View key metrics and browse your financial records.
            """)
            
            overview_btn = gr.Button("🔄 Refresh Data", variant="primary")
            
            overview_summary = gr.Markdown(label="Summary Statistics")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📄 Recent Invoices")
                    ar_table = gr.Dataframe(
                        label="Accounts Receivable",
                        height=400,
                        wrap=True
                    )
                
                with gr.Column():
                    gr.Markdown("### 💰 Recent Payments")
                    payments_table = gr.Dataframe(
                        label="Payments",
                        height=400,
                        wrap=True
                    )
            
            overview_btn.click(
                fn=get_data_overview,
                outputs=[overview_summary, ar_table, payments_table]
            )
        
        # ====================================================================
        # TAB 5: ANALYTICS
        # ====================================================================
        with gr.Tab("📈 Analytics", id="analytics"):
            gr.Markdown("""
            ## Financial Analytics Dashboard
            
            Visual insights into your financial data with interactive charts.
            """)
            
            analytics_btn = gr.Button("🔄 Refresh Charts", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    status_plot = gr.Plot(label="Invoice Status Distribution")
                with gr.Column():
                    customer_plot = gr.Plot(label="Top Customers")
            
            with gr.Row():
                budget_plot = gr.Plot(label="Budget vs Actual")
            
            with gr.Row():
                with gr.Column():
                    expense_category_plot = gr.Plot(label="Expenses by Category")
                with gr.Column():
                    expense_status_plot = gr.Plot(label="Expense Claim Status")
            
            analytics_btn.click(
                fn=refresh_analytics,
                outputs=[
                    status_plot,
                    customer_plot,
                    budget_plot,
                    expense_category_plot,
                    expense_status_plot
                ]
            )
        
        # ====================================================================
        # TAB 6: REPORTS
        # ====================================================================
        with gr.Tab("📄 Reports", id="reports"):
            gr.Markdown("""
            ## Generate Comprehensive Reports
            
            Create detailed financial analysis reports including all discrepancies,
            budget performance, and expense analysis.
            """)
            
            report_btn = gr.Button("📊 Generate Report", variant="primary", size="lg")
            
            report_output = gr.Textbox(
                label="Comprehensive Financial Report",
                lines=30,
                max_lines=50,
                show_copy_button=True
            )
            
            report_btn.click(fn=generate_report, outputs=report_output)
    
    # Footer
    gr.HTML("""
        <footer>
            <div style='margin-top: 3rem; padding-top: 2rem; border-top: 1px solid #e2e8f0;'>
                <p style='font-size: 0.95rem; color: #64748b;'>
                    Built with ❤️ using <strong>LangChain</strong>, <strong>ChromaDB</strong>, and <strong>Gradio</strong>
                </p>
                <p style='font-size: 0.85rem; color: #94a3b8; margin-top: 0.5rem;'>
                    Finance RAG Assistant v1.0.0 | 
                    <a href='https://github.com/yourusername/finance-rag-assistant' style='color: #667eea;'>GitHub</a>
                </p>
            </div>
        </footer>
    """)

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )