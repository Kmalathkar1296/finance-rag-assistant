import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain 1.1.0 imports for Python 3.12
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import Config

class FinanceRAGSystem:
    # RAG system for finance reconciliation
    
    def __init__(self, persist_directory=None):
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        
        # Initialize embeddings
        print("Loading embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Initialize vector store
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        
        # Data storage with type hints
        self.ar_df: Optional[pd.DataFrame] = None
        self.payments_df: Optional[pd.DataFrame] = None
        self.gl_df: Optional[pd.DataFrame] = None
        self.budget_df: Optional[pd.DataFrame] = None
        self.claims_df: Optional[pd.DataFrame] = None
        
    def load_data(self, ar_df: pd.DataFrame, payments_df: pd.DataFrame, 
                  gl_df: pd.DataFrame, budget_df: Optional[pd.DataFrame] = None,
                  claims_df: Optional[pd.DataFrame] = None) -> None:
        # Load financial data into the system
        self.ar_df = ar_df
        self.payments_df = payments_df
        self.gl_df = gl_df
        self.budget_df = budget_df
        self.claims_df = claims_df
        
        print(f"Loaded {len(ar_df)} AR records, {len(payments_df)} payments, "
              f"{len(gl_df)} GL entries")
        
        if budget_df is not None:
            print(f"Loaded {len(budget_df)} budget records")
        
        if claims_df is not None:
            print(f"Loaded {len(claims_df)} expense claims")
    
    def create_documents_for_embedding(self) -> List[str]:
        # Create text documents from financial data for embedding
        documents: List[str] = []
        
        # AR and Payment documents
        for _, ar in self.ar_df.iterrows():
            payment = self.payments_df[
                self.payments_df['ARID'] == ar['ARID']
            ].iloc[0] if not self.payments_df[
                self.payments_df['ARID'] == ar['ARID']
            ].empty else None
            
            doc_text = f"""
            Invoice ID: {ar['ARID']}
            Customer: {ar['Customer']}
            Invoice Date: {ar['InvoiceDate']}
            Due Date: {ar['DueDate']}
            Invoice Amount: ${ar['Amount']:.2f}
            Status: {ar['Status']}
            Payment Terms: {ar['Terms']}
            """
            
            if payment is not None:
                doc_text += f"""
            Payment ID: {payment['PaymentID']}
            Payment Date: {payment['PaymentDate']}
            Payment Amount: ${payment['Amount']:.2f}
            Payment Method: {payment['Method']}
            Reference: {payment['Reference']}
            Amount Difference: ${abs(ar['Amount'] - payment['Amount']):.2f}
            """
            else:
                doc_text += "\nNo payment record found for this invoice."
            
            # Add contextual analysis
            if payment is not None and abs(ar['Amount'] - payment['Amount']) > 0.01:
                doc_text += f"\nDISCREPANCY: Payment amount differs from invoice by ${abs(ar['Amount'] - payment['Amount']):.2f}"
            
            if ar['Status'] != 'Paid' and payment is None:
                due_date = datetime.strptime(ar['DueDate'], '%Y-%m-%d')
                if due_date < datetime.now():
                    days_overdue = (datetime.now() - due_date).days
                    doc_text += f"\nOVERDUE: Payment is {days_overdue} days overdue"
            
            documents.append(doc_text)
        
        # Budget documents
        if self.budget_df is not None:
            for _, budget in self.budget_df.iterrows():
                budget_text = f"""
            Budget Record
            Fiscal Year: {budget['FiscalYear']}
            Department: {budget['Dept']}
            Quarter: {budget['Quarter']}
            Budget: ${budget['BudgetUSD']:.2f}
            Forecast: ${budget['ForecastUSD']:.2f}
            Actual: ${budget['ActualUSD']:.2f}
            Variance: ${budget['VarianceUSD']:.2f}
            Notes: {budget['Notes']}
            Variance Percentage: {(budget['VarianceUSD'] / budget['BudgetUSD'] * 100):.1f}%
            """
                
                # Add variance analysis
                variance_pct = (budget['VarianceUSD'] / budget['BudgetUSD']) * 100
                if abs(variance_pct) > 10:
                    budget_text += f"\nSIGNIFICANT VARIANCE: {variance_pct:.1f}% difference from budget"
                
                documents.append(budget_text)
        
        # Expense Claims documents
        if self.claims_df is not None:
            for _, claim in self.claims_df.iterrows():
                claim_text = f"""
            Expense Claim
            Claim ID: {claim['ClaimID']}
            Employee ID: {claim['EmployeeID']}
            Submit Date: {claim['SubmitDate']}
            Category: {claim['Category']}
            Description: {claim['Description']}
            Amount: ${claim['Amount']:.2f}
            Currency: {claim['Currency']}
            Status: {claim['Status']}
            """
                
                if claim['ApprovedBy']:
                    claim_text += f"Approved By: {claim['ApprovedBy']}\n"
                
                if claim['PayDate']:
                    claim_text += f"Payment Date: {claim['PayDate']}\n"
                
                if claim['OverPolicyLimit']:
                    claim_text += "WARNING: This claim exceeds policy limits\n"
                
                # Add processing time analysis
                if claim['Status'] == 'Paid' and claim['PayDate']:
                    submit = datetime.strptime(claim['SubmitDate'], '%Y-%m-%d')
                    paid = datetime.strptime(claim['PayDate'], '%Y-%m-%d')
                    days_to_process = (paid - submit).days
                    claim_text += f"Processing Time: {days_to_process} days\n"
                    
                    if days_to_process > 14:
                        claim_text += "SLOW PROCESSING: Claim took longer than standard 14 days\n"
                
                documents.append(claim_text)
        
        return documents
    
    def build_vector_store(self) -> None:
        # Build ChromaDB vector store with LangChain 1.1.0
        print("Creating document embeddings...")
        documents = self.create_documents_for_embedding()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Create Document objects for latest LangChain
        doc_objects = [Document(page_content=doc) for doc in documents]
        splits = text_splitter.split_documents(doc_objects)
        
        # Create vector store with updated API
        print("Building vector store with ChromaDB...")
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="finance_data"
        )
        
        # Create retriever with updated parameters
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": Config.RETRIEVAL_K}
        )
        
        print(f"Vector store created with {len(splits)} document chunks")
    
    def find_discrepancies(self) -> List[Dict[str, Any]]:
        # Find all discrepancies in the data
        discrepancies: List[Dict[str, Any]] = []
        
        for _, ar in self.ar_df.iterrows():
            payment = self.payments_df[
                self.payments_df['ARID'] == ar['ARID']
            ]
            
            # Amount mismatch
            if not payment.empty:
                payment = payment.iloc[0]
                if abs(payment['Amount'] - ar['Amount']) > 0.01:
                    discrepancies.append({
                        'type': 'Amount Mismatch',
                        'severity': 'HIGH',
                        'invoice': ar['ARID'],
                        'customer': ar['Customer'],
                        'expected': ar['Amount'],
                        'received': payment['Amount'],
                        'difference': round(payment['Amount'] - ar['Amount'], 2)
                    })
            
            # Missing payment
            elif ar['Status'] == 'Paid':
                discrepancies.append({
                    'type': 'Missing Payment Record',
                    'severity': 'CRITICAL',
                    'invoice': ar['ARID'],
                    'customer': ar['Customer'],
                    'expected': ar['Amount'],
                    'received': 0,
                    'difference': -ar['Amount']
                })
            
            # Overdue payment
            if ar['Status'] != 'Paid':
                due_date = datetime.strptime(ar['DueDate'], '%Y-%m-%d')
                if due_date < datetime.now():
                    days_overdue = (datetime.now() - due_date).days
                    discrepancies.append({
                        'type': 'Overdue Payment',
                        'severity': 'CRITICAL' if days_overdue > 60 else 'MEDIUM',
                        'invoice': ar['ARID'],
                        'customer': ar['Customer'],
                        'expected': ar['Amount'],
                        'received': 0,
                        'difference': -ar['Amount'],
                        'days_overdue': days_overdue
                    })
        
        return discrepancies
    
    def query(self, question: str) -> Dict[str, Any]:
        # Query the RAG system
        if self.retriever is None:
            return {"error": "Vector store not initialized. Call build_vector_store() first."}
        
        # Retrieve relevant documents with updated API
        relevant_docs = self.retriever.invoke(question)
        
        # Analyze query intent
        question_lower = question.lower()
        
        response = {
            'question': question,
            'relevant_documents': [doc.page_content for doc in relevant_docs]
        }
        
        # Rule-based analysis
        if 'discrepanc' in question_lower or 'mismatch' in question_lower:
            discrepancies = self.find_discrepancies()
            response['discrepancies'] = discrepancies
            response['summary'] = f"Found {len(discrepancies)} discrepancies"
            response['analysis'] = self._analyze_discrepancies(discrepancies)
            
        elif 'overdue' in question_lower or 'late' in question_lower:
            overdue = self.ar_df[
                (self.ar_df['Status'] != 'Paid') & 
                (pd.to_datetime(self.ar_df['DueDate']) < datetime.now())
            ]
            response['overdue_invoices'] = overdue.to_dict('records')
            response['summary'] = f"Found {len(overdue)} overdue invoices totaling ${overdue['Amount'].sum():.2f}"
            
        elif 'customer' in question_lower:
            # Extract customer name from question
            customers = self.ar_df['Customer'].unique()
            for customer in customers:
                if customer.lower() in question_lower:
                    customer_data = self.ar_df[self.ar_df['Customer'] == customer]
                    response['customer_invoices'] = customer_data.to_dict('records')
                    response['summary'] = f"Found {len(customer_data)} invoices for {customer}"
                    break
        
        elif ('budget' in question_lower or 'variance' in question_lower) and self.budget_df is not None:
            significant_variances = self.budget_df[
                abs(self.budget_df['VarianceUSD'] / self.budget_df['BudgetUSD']) > 0.1
            ]
            response['budget_variances'] = significant_variances.to_dict('records')
            response['summary'] = f"Found {len(significant_variances)} departments with significant budget variances (>10%)"
            
            if 'department' in question_lower or 'dept' in question_lower:
                departments = self.budget_df['Dept'].unique()
                for dept in departments:
                    if dept.lower() in question_lower:
                        dept_budget = self.budget_df[self.budget_df['Dept'] == dept]
                        response['department_budget'] = dept_budget.to_dict('records')
                        total_variance = dept_budget['VarianceUSD'].sum()
                        response['summary'] = f"{dept} department: Total variance of ${total_variance:.2f}"
                        break
        
        elif ('expense' in question_lower or 'claim' in question_lower) and self.claims_df is not None:
            if 'pending' in question_lower or 'submitted' in question_lower:
                pending_claims = self.claims_df[self.claims_df['Status'] == 'Submitted']
                response['pending_claims'] = pending_claims.to_dict('records')
                response['summary'] = f"Found {len(pending_claims)} pending expense claims totaling ${pending_claims['Amount'].sum():.2f}"
            
            elif 'policy' in question_lower or 'over limit' in question_lower:
                over_limit = self.claims_df[self.claims_df['OverPolicyLimit'] == True]
                response['over_limit_claims'] = over_limit.to_dict('records')
                response['summary'] = f"Found {len(over_limit)} claims exceeding policy limits totaling ${over_limit['Amount'].sum():.2f}"
                response['recommendations'] = [
                    'Review policy limits with department managers',
                    'Investigate reasons for over-limit claims',
                    'Consider policy adjustments if limits are frequently exceeded'
                ]
            
            elif 'rejected' in question_lower:
                rejected = self.claims_df[self.claims_df['Status'] == 'Rejected']
                response['rejected_claims'] = rejected.to_dict('records')
                response['summary'] = f"Found {len(rejected)} rejected expense claims"
            
            else:
                category_summary = self.claims_df.groupby('Category')['Amount'].agg(['sum', 'count', 'mean'])
                response['expense_by_category'] = category_summary.to_dict()
                response['summary'] = f"Total expense claims: {len(self.claims_df)}, Total amount: ${self.claims_df['Amount'].sum():.2f}"
        
        else:
            response['summary'] = "I can help with discrepancies, overdue payments, customer queries, budget analysis, and expense claim management."
        
        return response
    
    def _analyze_discrepancies(self, discrepancies: List[Dict[str, Any]]) -> str:
        # Provide analysis of discrepancies
        if not discrepancies:
            return "No discrepancies found. All payments match invoices."
        
        critical = sum(1 for d in discrepancies if d['severity'] == 'CRITICAL')
        high = sum(1 for d in discrepancies if d['severity'] == 'HIGH')
        
        total_variance = sum(d['difference'] for d in discrepancies)
        
        analysis = f"""
        Analysis Summary:
        - Total Discrepancies: {len(discrepancies)}
        - Critical Issues: {critical}
        - High Priority: {high}
        - Total Variance: ${total_variance:.2f}
        
        Recommendations:
        1. Immediately investigate critical issues (missing payments, significant mismatches)
        2. Follow up with customers on overdue payments
        3. Review payment processing for amount mismatches
        4. Update ledger entries to reflect actual payments
        """
        
        return analysis
    
    def generate_report(self, output_file: str = "finance_report.txt") -> None:
        # Generate comprehensive reconciliation report
        discrepancies = self.find_discrepancies()
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE FINANCE REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # AR Summary
            f.write("ACCOUNTS RECEIVABLE SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Invoices: {len(self.ar_df)}\n")
            f.write(f"Total Payments: {len(self.payments_df)}\n")
            f.write(f"Total Outstanding: ${self.ar_df[self.ar_df['Status'] != 'Paid']['Amount'].sum():.2f}\n")
            f.write(f"Discrepancies Found: {len(discrepancies)}\n\n")
            
            # Budget summary
            if self.budget_df is not None:
                f.write("BUDGET PERFORMANCE SUMMARY\n")
                f.write("-" * 80 + "\n")
                total_budget = self.budget_df['BudgetUSD'].sum()
                total_actual = self.budget_df['ActualUSD'].sum()
                total_variance = self.budget_df['VarianceUSD'].sum()
                variance_pct = (total_variance / total_budget) * 100
                
                f.write(f"Total Budget: ${total_budget:,.2f}\n")
                f.write(f"Total Actual: ${total_actual:,.2f}\n")
                f.write(f"Total Variance: ${total_variance:,.2f} ({variance_pct:.1f}%)\n\n")
                
                f.write("Budget by Department:\n")
                dept_summary = self.budget_df.groupby('Dept').agg({
                    'BudgetUSD': 'sum',
                    'ActualUSD': 'sum',
                    'VarianceUSD': 'sum'
                })
                for dept, row in dept_summary.iterrows():
                    dept_var_pct = (row['VarianceUSD'] / row['BudgetUSD']) * 100
                    f.write(f"  {dept}: Budget ${row['BudgetUSD']:,.2f}, "
                           f"Actual ${row['ActualUSD']:,.2f}, "
                           f"Variance {dept_var_pct:+.1f}%\n")
                f.write("\n")
            
            # Expense claims summary
            if self.claims_df is not None:
                f.write("EXPENSE CLAIMS SUMMARY\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Claims: {len(self.claims_df)}\n")
                f.write(f"Total Amount: ${self.claims_df['Amount'].sum():,.2f}\n")
                
                status_summary = self.claims_df.groupby('Status').agg({
                    'ClaimID': 'count',
                    'Amount': 'sum'
                })
                f.write("\nBy Status:\n")
                for status, row in status_summary.iterrows():
                    f.write(f"  {status}: {row['ClaimID']} claims, ${row['Amount']:,.2f}\n")
                
                over_limit_count = self.claims_df['OverPolicyLimit'].sum()
                over_limit_amt = self.claims_df[self.claims_df['OverPolicyLimit'] == True]['Amount'].sum()
                f.write(f"\nOver Policy Limit: {over_limit_count} claims, ${over_limit_amt:,.2f}\n")
                
                category_summary = self.claims_df.groupby('Category')['Amount'].sum().sort_values(ascending=False)
                f.write("\nTop Expense Categories:\n")
                for category, amount in category_summary.head(5).items():
                    f.write(f"  {category}: ${amount:,.2f}\n")
                f.write("\n")
            
            # Discrepancies detail
            if discrepancies:
                f.write("PAYMENT DISCREPANCIES\n")
                f.write("-" * 80 + "\n")
                for disc in discrepancies:
                    f.write(f"\n[{disc['severity']}] {disc['type']}\n")
                    f.write(f"  Invoice: {disc['invoice']}\n")
                    f.write(f"  Customer: {disc['customer']}\n")
                    f.write(f"  Expected: ${disc['expected']:.2f}\n")
                    f.write(f"  Received: ${disc['received']:.2f}\n")
                    f.write(f"  Difference: ${disc['difference']:.2f}\n")
                    if 'days_overdue' in disc:
                        f.write(f"  Days Overdue: {disc['days_overdue']}\n")
            
            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n")
            
            recs = []
            if discrepancies:
                critical = sum(1 for d in discrepancies if d['severity'] == 'CRITICAL')
                if critical > 0:
                    recs.append(f"1. URGENT: Address {critical} critical payment discrepancies immediately")
            
            if self.budget_df is not None:
                over_budget_depts = self.budget_df[self.budget_df['VarianceUSD'] > 0].groupby('Dept')['VarianceUSD'].sum()
                if len(over_budget_depts) > 0:
                    recs.append(f"2. Review spending in {len(over_budget_depts)} departments over budget")
            
            if self.claims_df is not None:
                pending = len(self.claims_df[self.claims_df['Status'] == 'Submitted'])
                if pending > 0:
                    recs.append(f"3. Process {pending} pending expense claims")
                
                over_limit = self.claims_df['OverPolicyLimit'].sum()
                if over_limit > 0:
                    recs.append(f"4. Review {over_limit} claims exceeding policy limits")
            
            for rec in recs:
                f.write(f"{rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Comprehensive report generated: {output_file}")