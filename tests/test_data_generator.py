"""
Unit tests for the Finance Data Generator
Tests data generation functionality in isolation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
from datetime import datetime
from src.data_generator import FinanceDataGenerator


class TestDataGeneratorInitialization:
    """Test data generator initialization"""
    
    def test_initialization(self):
        """Test generator initializes with correct attributes"""
        gen = FinanceDataGenerator()
        
        assert gen is not None
        assert hasattr(gen, 'customers')
        assert hasattr(gen, 'departments')
        assert hasattr(gen, 'expense_categories')
        assert hasattr(gen, 'claim_statuses')
    
    def test_default_data_loaded(self):
        """Test default reference data is loaded"""
        gen = FinanceDataGenerator()
        
        assert len(gen.customers) > 0
        assert len(gen.departments) > 0
        assert len(gen.expense_categories) > 0
        assert len(gen.statuses) > 0


class TestAccountsReceivableGeneration:
    """Test AR data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
    
    def test_generate_basic_ar(self):
        """Test basic AR generation"""
        df = self.gen.generate_accounts_receivable(n=10)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
    
    def test_ar_columns(self):
        """Test AR has all required columns"""
        df = self.gen.generate_accounts_receivable(n=5)
        
        required_columns = [
            'ARID', 'Customer', 'InvoiceDate', 'DueDate',
            'Amount', 'Currency', 'Status', 'ReceivedDate', 'Terms'
        ]
        
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_ar_id_format(self):
        """Test AR IDs have correct format"""
        df = self.gen.generate_accounts_receivable(n=20)
        
        assert all(df['ARID'].str.startswith('AR'))
        assert all(df['ARID'].str.len() == 6)  # AR + 4 digits
    
    def test_ar_amounts_positive(self):
        """Test all amounts are positive"""
        df = self.gen.generate_accounts_receivable(n=50)
        
        assert all(df['Amount'] > 0)
        assert df['Amount'].min() >= 500
        assert df['Amount'].max() <= 5000
    
    def test_ar_dates_logical(self):
        """Test dates are in logical order"""
        df = self.gen.generate_accounts_receivable(n=10)
        
        for _, row in df.iterrows():
            invoice_date = pd.to_datetime(row['InvoiceDate'])
            due_date = pd.to_datetime(row['DueDate'])
            assert due_date > invoice_date
    
    def test_ar_status_values(self):
        """Test status values are valid"""
        df = self.gen.generate_accounts_receivable(n=100)
        
        valid_statuses = ['Paid', 'Pending', 'Overdue', 'Partial']
        assert all(df['Status'].isin(valid_statuses))
    
    def test_ar_paid_has_received_date(self):
        """Test paid invoices have received date"""
        df = self.gen.generate_accounts_receivable(n=100)
        
        paid = df[df['Status'] == 'Paid']
        assert all(pd.notna(paid['ReceivedDate']))
    
    def test_ar_unpaid_no_received_date(self):
        """Test unpaid invoices don't have received date"""
        df = self.gen.generate_accounts_receivable(n=100)
        
        unpaid = df[df['Status'] != 'Paid']
        assert all(pd.isna(unpaid['ReceivedDate']))


class TestPaymentsGeneration:
    """Test payment data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
        self.ar_df = self.gen.generate_accounts_receivable(n=20)
    
    def test_generate_payments(self):
        """Test basic payment generation"""
        df = self.gen.generate_payments(self.ar_df)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    
    def test_payment_columns(self):
        """Test payment has all required columns"""
        df = self.gen.generate_payments(self.ar_df)
        
        required_columns = [
            'PaymentID', 'ARID', 'PaymentDate', 'Amount',
            'Customer', 'Method', 'Reference'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_payment_references_valid_ar(self):
        """Test all payments reference valid AR records"""
        payments_df = self.gen.generate_payments(self.ar_df)
        
        assert all(payments_df['ARID'].isin(self.ar_df['ARID']))
    
    def test_payment_methods_valid(self):
        """Test payment methods are valid"""
        df = self.gen.generate_payments(self.ar_df)
        
        valid_methods = ['Wire', 'Check', 'ACH']
        assert all(df['Method'].isin(valid_methods))
    
    def test_payment_amounts_realistic(self):
        """Test payment amounts are realistic"""
        payments_df = self.gen.generate_payments(self.ar_df)
        
        # Payments should be close to invoice amounts
        for _, payment in payments_df.iterrows():
            ar_row = self.ar_df[self.ar_df['ARID'] == payment['ARID']].iloc[0]
            # Payment should be within 90-100% of invoice (some mismatches expected)
            assert 0.8 * ar_row['Amount'] <= payment['Amount'] <= 1.1 * ar_row['Amount']


class TestGeneralLedgerGeneration:
    """Test GL data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
        self.ar_df = self.gen.generate_accounts_receivable(n=15)
    
    def test_generate_gl(self):
        """Test basic GL generation"""
        df = self.gen.generate_general_ledger(self.ar_df)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(self.ar_df)
    
    def test_gl_columns(self):
        """Test GL has all required columns"""
        df = self.gen.generate_general_ledger(self.ar_df)
        
        required_columns = [
            'GLID', 'TxnDate', 'AccountNumber', 'AccountName',
            'Debit', 'Credit', 'Dept', 'CostCenter', 'Description', 'Currency'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_gl_amounts_match_ar(self):
        """Test GL amounts match AR amounts"""
        gl_df = self.gen.generate_general_ledger(self.ar_df)
        
        for i in range(len(self.ar_df)):
            assert gl_df.iloc[i]['Debit'] == self.ar_df.iloc[i]['Amount']
            assert gl_df.iloc[i]['Credit'] == 0.0


class TestBudgetGeneration:
    """Test budget forecast data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
    
    def test_generate_budget_one_year(self):
        """Test budget generation for one year"""
        df = self.gen.generate_budget_forecast(n_years=1)
        
        # 1 year * 6 departments * 4 quarters = 24
        assert len(df) == 24
    
    def test_generate_budget_multiple_years(self):
        """Test budget generation for multiple years"""
        df = self.gen.generate_budget_forecast(n_years=3)
        
        # 3 years * 6 departments * 4 quarters = 72
        assert len(df) == 72
    
    def test_budget_columns(self):
        """Test budget has all required columns"""
        df = self.gen.generate_budget_forecast(n_years=1)
        
        required_columns = [
            'FiscalYear', 'Dept', 'Quarter', 'BudgetUSD',
            'ForecastUSD', 'ActualUSD', 'VarianceUSD', 'Notes'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_budget_variance_calculation(self):
        """Test budget variance is calculated correctly"""
        df = self.gen.generate_budget_forecast(n_years=1)
        
        for _, row in df.iterrows():
            expected_variance = row['ActualUSD'] - row['BudgetUSD']
            assert abs(row['VarianceUSD'] - expected_variance) < 0.01
    
    def test_budget_all_departments(self):
        """Test budget includes all departments"""
        df = self.gen.generate_budget_forecast(n_years=1)
        
        departments = df['Dept'].unique()
        assert len(departments) == 6
    
    def test_budget_all_quarters(self):
        """Test budget includes all quarters"""
        df = self.gen.generate_budget_forecast(n_years=1)
        
        quarters = df['Quarter'].unique()
        assert len(quarters) == 4
        assert all(q in ['Q1', 'Q2', 'Q3', 'Q4'] for q in quarters)


class TestExpenseClaimsGeneration:
    """Test expense claims data generation"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
    
    def test_generate_claims(self):
        """Test basic claims generation"""
        df = self.gen.generate_expense_claims(n=50)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50
    
    def test_claims_columns(self):
        """Test claims has all required columns"""
        df = self.gen.generate_expense_claims(n=10)
        
        required_columns = [
            'ClaimID', 'EmployeeID', 'SubmitDate', 'Category',
            'Description', 'Amount', 'Currency', 'Status',
            'ApprovedBy', 'PayDate', 'OverPolicyLimit'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_claims_categories_valid(self):
        """Test expense categories are valid"""
        df = self.gen.generate_expense_claims(n=100)
        
        valid_categories = [
            'Travel', 'Meals', 'Supplies', 'Equipment',
            'Training', 'Software', 'Consulting'
        ]
        
        assert all(df['Category'].isin(valid_categories))
    
    def test_claims_status_valid(self):
        """Test claim statuses are valid"""
        df = self.gen.generate_expense_claims(n=100)
        
        valid_statuses = ['Submitted', 'Approved', 'Rejected', 'Paid']
        assert all(df['Status'].isin(valid_statuses))
    
    def test_claims_paid_has_date(self):
        """Test paid claims have payment date"""
        df = self.gen.generate_expense_claims(n=100)
        
        paid = df[df['Status'] == 'Paid']
        assert all(pd.notna(paid['PayDate']))
    
    def test_claims_approved_has_approver(self):
        """Test approved/paid claims have approver"""
        df = self.gen.generate_expense_claims(n=100)
        
        approved = df[df['Status'].isin(['Approved', 'Paid'])]
        assert all(pd.notna(approved['ApprovedBy']))
    
    def test_claims_amounts_by_category(self):
        """Test expense amounts are realistic by category"""
        df = self.gen.generate_expense_claims(n=200)
        
        # Travel should generally be more expensive than meals
        travel = df[df['Category'] == 'Travel']['Amount'].mean()
        meals = df[df['Category'] == 'Meals']['Amount'].mean()
        
        assert travel > meals


class TestHelperMethods:
    """Test helper methods in data generator"""
    
    def setup_method(self):
        """Setup for each test"""
        self.gen = FinanceDataGenerator()
    
    def test_expense_amount_ranges(self):
        """Test expense amounts are within expected ranges"""
        # Test multiple categories
        categories = ['Travel', 'Meals', 'Equipment']
        
        for category in categories:
            amount = self.gen._get_expense_amount(category)
            assert amount > 0
            assert amount < 10000  # Reasonable upper bound
    
    def test_expense_description_generation(self):
        """Test expense descriptions are generated"""
        categories = ['Travel', 'Meals', 'Software']
        
        for category in categories:
            desc = self.gen._generate_expense_description(category)
            assert isinstance(desc, str)
            assert len(desc) > 0
    
    def test_budget_note_generation(self):
        """Test budget notes are meaningful"""
        # Over budget
        note = self.gen._generate_budget_note(15000, 10000)
        assert 'over budget' in note.lower()
        
        # Under budget
        note = self.gen._generate_budget_note(5000, 10000)
        assert 'under budget' in note.lower()
        
        # Within range
        note = self.gen._generate_budget_note(10500, 10000)
        assert 'acceptable' in note.lower()


# Run specific tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])