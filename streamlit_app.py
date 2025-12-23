"""
Finance RAG Assistant - Streamlit Web Application
Deploy this app to Streamlit Cloud for free public access
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Import local modules
from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Finance RAG Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - CONFIGURATION
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/financial-analytics.png", width=150)
    st.title("⚙️ Configuration")
    
    # API Key Status
    st.markdown("### 🔑 API Configuration")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if api_key:
        st.success("✅ API Key configured")
        use_llm = st.checkbox("Use AI-powered responses", value=False, 
                              help="Enable for smarter analysis (requires API key)")
    else:
        st.info("ℹ️ Using rule-based analysis (no API key needed)")
        use_llm = False
    
    st.markdown("---")
    
    # Data Generation Settings
    st.markdown("### 📊 Data Settings")
    n_invoices = st.slider("Number of Invoices", min_value=10, max_value=200, value=50, step=10)
    n_claims = st.slider("Number of Expense Claims", min_value=20, max_value=400, value=100, step=20)
    
    if st.button("🎲 Generate Sample Data", type="primary"):
        with st.spinner("Generating financial data..."):
            try:
                gen = FinanceDataGenerator()
                
                # Generate data
                st.session_state.ar_df = gen.generate_accounts_receivable(n=n_invoices)
                st.session_state.payments_df = gen.generate_payments(st.session_state.ar_df)
                st.session_state.gl_df = gen.generate_general_ledger(st.session_state.ar_df)
                st.session_state.budget_df = gen.generate_budget_forecast(n_years=1)
                st.session_state.claims_df = gen.generate_expense_claims(n=n_claims)
                
                # Build RAG system
                st.session_state.rag_system = FinanceRAGSystem()
                st.session_state.rag_system.load_data(
                    st.session_state.ar_df,
                    st.session_state.payments_df,
                    st.session_state.gl_df,
                    st.session_state.budget_df,
                    st.session_state.claims_df
                )
                st.session_state.rag_system.build_vector_store()
                st.session_state.data_generated = True
                
                st.success(f"✅ Generated {n_invoices} invoices and {n_claims} claims!")
                st.balloons()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    
    # About Section
    st.markdown("### 📖 About")
    st.markdown("""
    **Finance RAG Assistant** uses AI to analyze financial data:
    - 🔍 Payment reconciliation
    - 📊 Budget variance analysis  
    - 💳 Expense management
    - 🤖 Natural language queries
    """)
    
    st.markdown("---")
    st.caption("Built with LangChain, ChromaDB & Streamlit")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<h1 class="main-header">💰 Finance RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown("**AI-Powered Payment Reconciliation & Financial Analysis**")
st.markdown("---")

# Check if data is generated
if 'data_generated' not in st.session_state or not st.session_state.data_generated:
    st.info("👈 **Get Started:** Generate sample data using the sidebar!")
    
    # Show demo information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 Smart Analysis")
        st.write("Automatically detect payment discrepancies and anomalies")
    
    with col2:
        st.markdown("### 📊 Real-Time Insights")
        st.write("Query your financial data in natural language")
    
    with col3:
        st.markdown("### 💡 Actionable Reports")
        st.write("Get recommendations based on your data")
    
    st.stop()

# ============================================================================
# TABS - MAIN FEATURES
# ============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Query Assistant", 
    "📊 Data Overview", 
    "📈 Analytics", 
    "⚠️ Discrepancies",
    "📄 Reports"
])

# ----------------------------------------------------------------------------
# TAB 1: QUERY ASSISTANT
# ----------------------------------------------------------------------------

with tab1:
    st.header("🔍 Ask Questions About Your Financial Data")
    
    # Sample queries as buttons
    st.markdown("### Quick Queries")
    col1, col2, col3, col4 = st.columns(4)
    
    query = None
    with col1:
        if st.button("💸 Payment Discrepancies", use_container_width=True):
            query = "Show me all payment discrepancies"
    with col2:
        if st.button("⏰ Overdue Payments", use_container_width=True):
            query = "Which payments are overdue?"
    with col3:
        if st.button("📊 Budget Variance", use_container_width=True):
            query = "Which departments are over budget?"
    with col4:
        if st.button("💳 Pending Claims", use_container_width=True):
            query = "Show me pending expense claims"
    
    # Custom query input
    custom_query = st.text_input(
        "Or type your own question:", 
        placeholder="e.g., Show me invoices for Acme Corp",
        key="custom_query"
    )
    
    if custom_query:
        query = custom_query
    
    if query or st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("🤔 Analyzing your query..."):
                try:
                    result = st.session_state.rag_system.query(query)
                    
                    # Display results
                    st.markdown("### 📊 Results")
                    st.info(result.get('summary', 'No summary available'))
                    
                    # Analysis
                    if 'analysis' in result:
                        with st.expander("📈 Detailed Analysis", expanded=True):
                            st.text(result['analysis'])
                    
                    # Recommendations
                    if 'recommendations' in result:
                        with st.expander("💡 Recommendations"):
                            for rec in result['recommendations']:
                                st.markdown(f"- {rec}")
                    
                    # Data tables
                    if 'discrepancies' in result and result['discrepancies']:
                        with st.expander("⚠️ Discrepancy Details"):
                            df = pd.DataFrame(result['discrepancies'])
                            st.dataframe(df, use_container_width=True)
                    
                    if 'overdue_invoices' in result and result['overdue_invoices']:
                        with st.expander("⏰ Overdue Invoice Details"):
                            df = pd.DataFrame(result['overdue_invoices'])
                            st.dataframe(df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

# ----------------------------------------------------------------------------
# TAB 2: DATA OVERVIEW
# ----------------------------------------------------------------------------

with tab2:
    st.header("📊 Financial Data Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_invoices = len(st.session_state.ar_df)
        st.metric("Total Invoices", total_invoices)
    
    with col2:
        total_payments = len(st.session_state.payments_df)
        st.metric("Payments Recorded", total_payments)
    
    with col3:
        outstanding = st.session_state.ar_df[
            st.session_state.ar_df['Status'] != 'Paid'
        ]['Amount'].sum()
        st.metric("Outstanding", f"${outstanding:,.2f}")
    
    with col4:
        collected = st.session_state.ar_df[
            st.session_state.ar_df['Status'] == 'Paid'
        ]['Amount'].sum()
        st.metric("Collected", f"${collected:,.2f}")
    
    st.markdown("---")
    
    # Data tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📄 Recent Invoices")
        st.dataframe(
            st.session_state.ar_df.head(10),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.markdown("### 💰 Recent Payments")
        st.dataframe(
            st.session_state.payments_df.head(10),
            use_container_width=True,
            hide_index=True
        )
    
    # Budget overview
    if st.session_state.budget_df is not None:
        st.markdown("---")
        st.markdown("### 💼 Budget Overview")
        
        budget_summary = st.session_state.budget_df.groupby('Dept').agg({
            'BudgetUSD': 'sum',
            'ActualUSD': 'sum',
            'VarianceUSD': 'sum'
        }).reset_index()
        
        st.dataframe(budget_summary, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------------
# TAB 3: ANALYTICS
# ----------------------------------------------------------------------------

with tab3:
    st.header("📈 Financial Analytics Dashboard")
    
    # Invoice status distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Invoice Status Distribution")
        status_counts = st.session_state.ar_df['Status'].value_counts()
        fig = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title="Invoice Status",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💰 Amount by Customer")
        customer_amounts = st.session_state.ar_df.groupby('Customer')['Amount'].sum().sort_values(ascending=False).head(8)
        fig = px.bar(
            x=customer_amounts.index,
            y=customer_amounts.values,
            title="Top 8 Customers by Invoice Amount",
            labels={'x': 'Customer', 'y': 'Amount ($)'},
            color=customer_amounts.values,
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Budget analysis
    if st.session_state.budget_df is not None:
        st.markdown("---")
        st.markdown("### 📊 Budget vs Actual by Department")
        
        budget_by_dept = st.session_state.budget_df.groupby('Dept').agg({
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
        fig.update_layout(barmode='group', title="Budget vs Actual Spending")
        st.plotly_chart(fig, use_container_width=True)
    
    # Expense claims analysis
    if st.session_state.claims_df is not None:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 💳 Expense Claims by Category")
            category_amounts = st.session_state.claims_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
            fig = px.bar(
                x=category_amounts.values,
                y=category_amounts.index,
                orientation='h',
                title="Spending by Category",
                labels={'x': 'Amount ($)', 'y': 'Category'},
                color=category_amounts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📋 Claims Status")
            status_counts = st.session_state.claims_df['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Expense Claim Status",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# TAB 4: DISCREPANCIES
# ----------------------------------------------------------------------------

with tab4:
    st.header("⚠️ Payment Discrepancies & Issues")
    
    # Find discrepancies
    with st.spinner("Analyzing for discrepancies..."):
        discrepancies = st.session_state.rag_system.find_discrepancies()
    
    if discrepancies:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Discrepancies", len(discrepancies))
        with col2:
            critical = sum(1 for d in discrepancies if d['severity'] == 'CRITICAL')
            st.metric("Critical Issues", critical, delta=None if critical == 0 else "Needs attention")
        with col3:
            total_variance = sum(d['difference'] for d in discrepancies)
            st.metric("Total Variance", f"${total_variance:,.2f}")
        
        st.markdown("---")
        
        # Discrepancy table
        df_discrepancies = pd.DataFrame(discrepancies)
        
        # Color code by severity
        def color_severity(val):
            if val == 'CRITICAL':
                return 'background-color: #ff4444; color: white'
            elif val == 'HIGH':
                return 'background-color: #ffaa00; color: white'
            elif val == 'MEDIUM':
                return 'background-color: #ffdd44'
            return ''
        
        styled_df = df_discrepancies.style.applymap(color_severity, subset=['severity'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export option
        csv = df_discrepancies.to_csv(index=False)
        st.download_button(
            label="📥 Download Discrepancies CSV",
            data=csv,
            file_name=f"discrepancies_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ No discrepancies found! All payments match invoices.")

# ----------------------------------------------------------------------------
# TAB 5: REPORTS
# ----------------------------------------------------------------------------

with tab5:
    st.header("📄 Generate Financial Reports")
    
    st.markdown("""
    Generate comprehensive reports for:
    - Payment reconciliation
    - Budget performance
    - Expense analysis
    """)
    
    if st.button("📊 Generate Comprehensive Report", type="primary"):
        with st.spinner("Generating report..."):
            import tempfile
            
            # Generate report
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.txt')
            st.session_state.rag_system.generate_report(temp_file.name)
            
            # Read and display
            with open(temp_file.name, 'r') as f:
                report_content = f.read()
            
            st.text_area("Report Preview", report_content, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download Full Report",
                data=report_content,
                file_name=f"finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("✅ Report generated successfully!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using <strong>LangChain</strong>, <strong>ChromaDB</strong>, and <strong>Streamlit</strong></p>
        <p style='font-size: 0.8rem; color: gray'>
            Finance RAG Assistant v1.0.0 | 
            <a href='https://github.com/yourusername/finance-rag-assistant'>GitHub</a>
        </p>
    </div>
    """, unsafe_allow_html=True)