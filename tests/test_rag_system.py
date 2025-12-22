"""
Unit tests for the Finance RAG System
Tests core functionality without requiring API keys
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil

from src.data_generator import FinanceDataGenerator
from src.rag_system import FinanceRAGSystem


class TestFinanceDataGenerator:
    """Test the data generation functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = FinanceDataGenerator()
    
    def test_generator_initialization(self):
        """Test that generator initializes correctly"""
        assert self.generator is not None
        assert len(self.generator.customers) > 0
        assert len(self.generator.departments) > 0
    
    def test_generate_ar_basic(self):
        """Test basic AR generation"""
        ar_df = self.generator.generate_accounts_receivable(n=10)
        
        assert len(ar_df) == 10
        assert 'ARID' in ar_df.columns
        assert 'Customer' in ar_df.columns
        assert 'Amount' in ar_df.columns
        assert 'Status' in ar_df.columns
    
    def test_generate_ar_data_types(self):
        """Test AR data types are correct"""
        ar_df = self.generator.generate_accounts_receivable(n=5)
        
        assert ar_df['Amount'].dtype in ['float64', 'float32']
        assert all(ar_df['Amount'] > 0)
        assert all(ar_df['Status'].isin(['Paid', 'Pending', 'Overdue', 'Partial']))
    
    def test_generate_ar_unique_ids(self):
        """Test that AR IDs are unique"""
        ar_df = self.generator.generate_accounts_receivable(n=20)
        
        assert len(ar_df['ARID'].unique()) == 20
        assert all(ar_df['ARID'].str.startswith('AR'))
    
    def test_generate_payments(self):
        """Test payment generation"""
        ar_df = self.generator.generate_accounts_receivable(n=10)
        payments_df = self.generator.generate_payments(ar_df)
        
        assert len(payments_df) > 0
        assert 'PaymentID' in payments_df.columns
        assert 'ARID' in payments_df.columns
        assert 'Amount' in payments_df.columns
        
        # Check that payment ARIDs reference valid AR records
        assert all(payments_df['ARID'].isin(ar_df['ARID']))
    
    def test_generate_gl(self):
        """Test general ledger generation"""
        ar_df = self.generator.generate_accounts_receivable(n=10)
        gl_df = self.generator.generate_general_ledger(ar_df)
        
        assert len(gl_df) == len(ar_df)
        assert 'GLID' in gl_df.columns
        assert 'AccountNumber' in gl_df.columns
        assert 'Debit' in gl_df.columns
    
    def test_generate_budget(self):
        """Test budget forecast generation"""
        budget_df = self.generator.generate_budget_forecast(n_years=1)
        
        # 1 year * 6 departments * 4 quarters = 24 records
        assert len(budget_df) == 24
        assert 'FiscalYear' in budget_df.columns
        assert 'Dept' in budget_df.columns
        assert 'BudgetUSD' in budget_df.columns
        assert 'ActualUSD' in budget_df.columns
        assert 'VarianceUSD' in budget_df.columns
    
    def test_generate_expense_claims(self):
        """Test expense claims generation"""
        claims_df = self.generator.generate_expense_claims(n=50)
        
        assert len(claims_df) == 50
        assert 'ClaimID' in claims_df.columns
        assert 'Category' in claims_df.columns
        assert 'Amount' in claims_df.columns
        assert 'Status' in claims_df.columns
        
        # Check that categories are valid
        valid_categories = ['Travel', 'Meals', 'Supplies', 'Equipment', 
                          'Training', 'Software', 'Consulting']
        assert all(claims_df['Category'].isin(valid_categories))
    
    def test_expense_amounts_realistic(self):
        """Test that expense amounts are within realistic ranges"""
        claims_df = self.generator.generate_expense_claims(n=100)
        
        # Check reasonable bounds (accounting for over-limit multiplier)
        assert claims_df['Amount'].min() > 10  # Minimum reasonable
        assert claims_df['Amount'].max() < 10000  # Maximum reasonable
        assert claims_df['Amount'].mean() > 100  # Reasonable average


class TestFinanceRAGSystem:
    """Test the RAG system functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.rag = FinanceRAGSystem(persist_directory=self.temp_dir)
        
        # Generate test data
        self.generator = FinanceDataGenerator()
        self.ar_df = self.generator.generate_accounts_receivable(n=20)
        self.payments_df = self.generator.generate_payments(self.ar_df)
        self.gl_df = self.generator.generate_general_ledger(self.ar_df)
        self.budget_df = self.generator.generate_budget_forecast(n_years=1)
        self.claims_df = self.generator.generate_expense_claims(n=30)
    
    def teardown_method(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_rag_initialization(self):
        """Test RAG system initializes correctly"""
        assert self.rag is not None
        assert self.rag.embeddings is not None
        assert self.rag.persist_directory == self.temp_dir
    
    def test_load_data(self):
        """Test data loading"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df,
            self.claims_df
        )
        
        assert self.rag.ar_df is not None
        assert len(self.rag.ar_df) == 20
        assert self.rag.payments_df is not None
        assert self.rag.budget_df is not None
        assert self.rag.claims_df is not None
    
    def test_create_documents(self):
        """Test document creation for embedding"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df,
            self.claims_df
        )
        
        documents = self.rag.create_documents_for_embedding()
        
        assert len(documents) > 0
        assert isinstance(documents, list)
        assert all(isinstance(doc, str) for doc in documents)
        
        # Should have docs for AR + budget + claims
        expected_min = len(self.ar_df) + len(self.budget_df) + len(self.claims_df)
        assert len(documents) >= expected_min
    
    def test_build_vector_store(self):
        """Test vector store creation"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df,
            self.claims_df
        )
        
        self.rag.build_vector_store()
        
        assert self.rag.vectorstore is not None
        assert self.rag.retriever is not None
    
    def test_find_discrepancies(self):
        """Test discrepancy detection"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df
        )
        
        discrepancies = self.rag.find_discrepancies()
        
        assert isinstance(discrepancies, list)
        # Should find at least some discrepancies in generated data
        assert len(discrepancies) > 0
        
        # Check discrepancy structure
        if discrepancies:
            disc = discrepancies[0]
            assert 'type' in disc
            assert 'severity' in disc
            assert 'invoice' in disc
            assert 'customer' in disc
    
    def test_query_without_vectorstore(self):
        """Test query fails gracefully without vector store"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df
        )
        
        result = self.rag.query("test query")
        
        assert 'error' in result
        assert 'not initialized' in result['error'].lower()
    
    def test_query_with_vectorstore(self):
        """Test query works with vector store"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df,
            self.claims_df
        )
        
        self.rag.build_vector_store()
        
        result = self.rag.query("Show me all discrepancies")
        
        assert 'question' in result
        assert 'summary' in result
        assert result['question'] == "Show me all discrepancies"
    
    def test_query_overdue_payments(self):
        """Test query for overdue payments"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df
        )
        
        self.rag.build_vector_store()
        
        result = self.rag.query("Which payments are overdue?")
        
        assert 'summary' in result
        assert 'overdue' in result['summary'].lower()
    
    def test_query_budget_variance(self):
        """Test query for budget variances"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df
        )
        
        self.rag.build_vector_store()
        
        result = self.rag.query("Show me budget variances")
        
        assert 'summary' in result
        assert 'budget' in result['summary'].lower() or 'variance' in result['summary'].lower()
    
    def test_query_expense_claims(self):
        """Test query for expense claims"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            budget_df=None,
            claims_df=self.claims_df
        )
        
        self.rag.build_vector_store()
        
        result = self.rag.query("Show me pending expense claims")
        
        assert 'summary' in result
    
    def test_generate_report(self):
        """Test report generation"""
        self.rag.load_data(
            self.ar_df,
            self.payments_df,
            self.gl_df,
            self.budget_df,
            self.claims_df
        )
        
        report_file = os.path.join(self.temp_dir, "test_report.txt")
        self.rag.generate_report(report_file)
        
        assert os.path.exists(report_file)
        
        # Check report content
        with open(report_file, 'r') as f:
            content = f.read()
        
        assert 'COMPREHENSIVE FINANCE REPORT' in content
        assert 'ACCOUNTS RECEIVABLE SUMMARY' in content
        assert 'Total Invoices' in content


class TestDataIntegrity:
    """Test data integrity and relationships"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.generator = FinanceDataGenerator()
    
    def test_payment_ar_relationship(self):
        """Test that all payments reference valid AR records"""
        ar_df = self.generator.generate_accounts_receivable(n=50)
        payments_df = self.generator.generate_payments(ar_df)
        
        # All payment ARIDs should be in AR
        assert all(payments_df['ARID'].isin(ar_df['ARID']))
    
    def test_gl_ar_relationship(self):
        """Test GL entries match AR records"""
        ar_df = self.generator.generate_accounts_receivable(n=20)
        gl_df = self.generator.generate_general_ledger(ar_df)
        
        # Should have same number of records
        assert len(gl_df) == len(ar_df)
        
        # Amounts should match
        for i, ar_row in ar_df.iterrows():
            gl_row = gl_df.iloc[i]
            assert gl_row['Debit'] == ar_row['Amount']
    
    def test_budget_structure(self):
        """Test budget data has correct structure"""
        budget_df = self.generator.generate_budget_forecast(n_years=2)
        
        # Should have all departments
        departments = budget_df['Dept'].unique()
        assert len(departments) == 6
        
        # Should have all quarters for each dept/year
        for dept in departments:
            dept_data = budget_df[budget_df['Dept'] == dept]
            assert len(dept_data) == 8  # 2 years * 4 quarters
    
    def test_expense_claims_consistency(self):
        """Test expense claims data consistency"""
        claims_df = self.generator.generate_expense_claims(n=100)
        
        # Paid claims should have payment date
        paid_claims = claims_df[claims_df['Status'] == 'Paid']
        assert all(pd.notna(paid_claims['PayDate']))
        
        # Approved/Paid claims should have approver
        approved_claims = claims_df[claims_df['Status'].isin(['Approved', 'Paid'])]
        assert all(pd.notna(approved_claims['ApprovedBy']))


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data(self):
        """Test system handles empty data gracefully"""
        generator = FinanceDataGenerator()
        ar_df = generator.generate_accounts_receivable(n=0)
        
        assert len(ar_df) == 0
        assert isinstance(ar_df, pd.DataFrame)
    
    def test_large_dataset(self):
        """Test system handles large datasets"""
        generator = FinanceDataGenerator()
        ar_df = generator.generate_accounts_receivable(n=1000)
        
        assert len(ar_df) == 1000
        assert len(ar_df['ARID'].unique()) == 1000
    
    def test_rag_without_budget(self):
        """Test RAG works without budget data"""
        rag = FinanceRAGSystem(persist_directory=tempfile.mkdtemp())
        generator = FinanceDataGenerator()
        
        ar_df = generator.generate_accounts_receivable(n=10)
        payments_df = generator.generate_payments(ar_df)
        gl_df = generator.generate_general_ledger(ar_df)
        
        # Load without budget or claims
        rag.load_data(ar_df, payments_df, gl_df)
        docs = rag.create_documents_for_embedding()
        
        assert len(docs) > 0
    
    def test_rag_without_claims(self):
        """Test RAG works without expense claims"""
        rag = FinanceRAGSystem(persist_directory=tempfile.mkdtemp())
        generator = FinanceDataGenerator()
        
        ar_df = generator.generate_accounts_receivable(n=10)
        payments_df = generator.generate_payments(ar_df)
        gl_df = generator.generate_general_ledger(ar_df)
        budget_df = generator.generate_budget_forecast(n_years=1)
        
        # Load without claims
        rag.load_data(ar_df, payments_df, gl_df, budget_df, None)
        docs = rag.create_documents_for_embedding()
        
        assert len(docs) > 0


# Pytest fixtures
@pytest.fixture
def sample_ar_data():
    """Fixture for sample AR data"""
    generator = FinanceDataGenerator()
    return generator.generate_accounts_receivable(n=10)

@pytest.fixture
def sample_budget_data():
    """Fixture for sample budget data"""
    generator = FinanceDataGenerator()
    return generator.generate_budget_forecast(n_years=1)

@pytest.fixture
def sample_claims_data():
    """Fixture for sample claims data"""
    generator = FinanceDataGenerator()
    return generator.generate_expense_claims(n=20)


# Run tests
if __name__ == "__main__":
    # Run with: python -m pytest tests/test_rag_system.py -v
    pytest.main([__file__, "-v", "--tb=short"])