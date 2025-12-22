import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinanceDataGenerator:
    # Generate synthetic financial data similar to Excelx.com datasets
    
    def __init__(self):
        self.customers = ['Acme Corp', 'TechStart Inc', 'Global Solutions', 
                         'Innovate Ltd', 'Prime Retail', 'BlueSky Industries',
                         'NextGen Systems', 'Alpha Enterprises']
        self.statuses = ['Paid', 'Pending', 'Overdue', 'Partial']
        self.departments = ['Sales', 'Marketing', 'IT', 'Operations', 'HR', 'Finance']
        self.expense_categories = ['Travel', 'Meals', 'Supplies', 'Equipment', 
                                   'Training', 'Software', 'Consulting']
        self.claim_statuses = ['Submitted', 'Approved', 'Rejected', 'Paid']
        
    def generate_accounts_receivable(self, n=100):
        # Generate Accounts Receivable data
        data = []
        for i in range(1, n + 1):
            invoice_date = datetime(2024, np.random.randint(1, 13), 
                                   np.random.randint(1, 29))
            due_date = invoice_date + timedelta(days=30)
            amount = round(np.random.uniform(500, 5000), 2)
            customer = np.random.choice(self.customers)
            status = np.random.choice(self.statuses, p=[0.5, 0.2, 0.2, 0.1])
            
            received_date = None
            if status == 'Paid':
                received_date = (due_date + timedelta(
                    days=np.random.randint(-5, 10))).strftime('%Y-%m-%d')
            
            data.append({
                'ARID': f'AR{str(i).zfill(4)}',
                'Customer': customer,
                'InvoiceDate': invoice_date.strftime('%Y-%m-%d'),
                'DueDate': due_date.strftime('%Y-%m-%d'),
                'Amount': amount,
                'Currency': 'USD',
                'Status': status,
                'ReceivedDate': received_date,
                'Terms': 'Net 30'
            })
        
        return pd.DataFrame(data)
    
    def generate_payments(self, ar_df):
        # Generate payment records with intentional mismatches
        data = []
        payment_id = 1
        
        for _, ar in ar_df.iterrows():
            # 80% of paid invoices have payments, introduce some mismatches
            if ar['Status'] == 'Paid' or np.random.random() > 0.7:
                # Introduce amount mismatches in 15% of cases
                if np.random.random() < 0.15:
                    payment_amount = round(ar['Amount'] * np.random.uniform(0.9, 0.99), 2)
                else:
                    payment_amount = ar['Amount']
                
                payment_date = ar['ReceivedDate'] if ar['ReceivedDate'] else \
                               (datetime.strptime(ar['DueDate'], '%Y-%m-%d') + 
                                timedelta(days=np.random.randint(0, 10))).strftime('%Y-%m-%d')
                
                data.append({
                    'PaymentID': f'PAY{str(payment_id).zfill(4)}',
                    'ARID': ar['ARID'],
                    'PaymentDate': payment_date,
                    'Amount': payment_amount,
                    'Customer': ar['Customer'],
                    'Method': np.random.choice(['Wire', 'Check', 'ACH']),
                    'Reference': f'REF{np.random.randint(10000, 99999)}'
                })
                payment_id += 1
        
        return pd.DataFrame(data)
    
    def generate_general_ledger(self, ar_df):
        # Generate General Ledger entries
        data = []
        
        for idx, ar in ar_df.iterrows():
            data.append({
                'GLID': f'GL{str(idx + 1).zfill(4)}',
                'TxnDate': ar['InvoiceDate'],
                'AccountNumber': '1200',
                'AccountName': 'Accounts Receivable',
                'Debit': ar['Amount'],
                'Credit': 0.0,
                'Dept': 'Sales',
                'CostCenter': f'CC{np.random.randint(1, 5):02d}',
                'Description': f'Invoice {ar["Customer"]}',
                'Currency': 'USD'
            })
        
        return pd.DataFrame(data)
    
    def generate_budget_forecast(self, n_years=2):
        # Generate Budget Forecast data
        data = []
        current_year = 2024
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        
        for year in range(current_year, current_year + n_years):
            for dept in self.departments:
                for quarter in quarters:
                    # Generate realistic budget figures
                    budget_base = np.random.uniform(50000, 200000)
                    forecast = budget_base * np.random.uniform(0.95, 1.1)
                    actual = forecast * np.random.uniform(0.9, 1.15)
                    variance = actual - budget_base
                    
                    data.append({
                        'FiscalYear': year,
                        'Dept': dept,
                        'Quarter': quarter,
                        'BudgetUSD': round(budget_base, 2),
                        'ForecastUSD': round(forecast, 2),
                        'ActualUSD': round(actual, 2),
                        'VarianceUSD': round(variance, 2),
                        'Notes': self._generate_budget_note(variance, budget_base)
                    })
        
        return pd.DataFrame(data)
    
    def _generate_budget_note(self, variance, budget):
        # Generate contextual notes for budget variances
        variance_pct = (variance / budget) * 100
        
        if variance_pct > 10:
            return f"Over budget by {variance_pct:.1f}% - investigate spending"
        elif variance_pct < -10:
            return f"Under budget by {abs(variance_pct):.1f}% - strong cost control"
        else:
            return "Within acceptable variance range"
    
    def generate_expense_claims(self, n=200):
        # Generate Expense Claims data
        data = []
        
        for i in range(1, n + 1):
            submit_date = datetime(2024, np.random.randint(1, 13), 
                                   np.random.randint(1, 29))
            
            # Determine claim status and dates
            status = np.random.choice(self.claim_statuses, 
                                     p=[0.15, 0.50, 0.10, 0.25])
            
            approved_by = None
            pay_date = None
            
            if status in ['Approved', 'Paid']:
                approved_by = f'MGR{np.random.randint(1, 10):03d}'
                
            if status == 'Paid':
                pay_date = (submit_date + timedelta(
                    days=np.random.randint(7, 21))).strftime('%Y-%m-%d')
            
            # Generate realistic amounts by category
            category = np.random.choice(self.expense_categories)
            amount = self._get_expense_amount(category)
            
            # Occasionally create claims over policy limit
            over_limit = np.random.random() < 0.1
            if over_limit:
                amount *= 1.5
            
            data.append({
                'ClaimID': f'CLM{str(i).zfill(4)}',
                'EmployeeID': f'EMP{np.random.randint(1, 50):03d}',
                'SubmitDate': submit_date.strftime('%Y-%m-%d'),
                'Category': category,
                'Description': self._generate_expense_description(category),
                'Amount': round(amount, 2),
                'Currency': 'USD',
                'Status': status,
                'ApprovedBy': approved_by,
                'PayDate': pay_date,
                'OverPolicyLimit': over_limit
            })
        
        return pd.DataFrame(data)
    
    def _get_expense_amount(self, category):
        # Generate realistic amounts based on expense category
        amount_ranges = {
            'Travel': (200, 2000),
            'Meals': (20, 150),
            'Supplies': (50, 500),
            'Equipment': (500, 3000),
            'Training': (300, 2500),
            'Software': (100, 1000),
            'Consulting': (1000, 5000)
        }
        
        min_amt, max_amt = amount_ranges.get(category, (50, 500))
        return np.random.uniform(min_amt, max_amt)
    
    def _generate_expense_description(self, category):
        # Generate realistic expense descriptions
        descriptions = {
            'Travel': ['Flight to client site', 'Hotel accommodation', 'Rental car', 
                      'Train ticket', 'Taxi fare'],
            'Meals': ['Client dinner', 'Team lunch', 'Conference meals', 
                     'Business breakfast'],
            'Supplies': ['Office supplies', 'Printer cartridges', 'Stationery', 
                        'Cleaning supplies'],
            'Equipment': ['Laptop computer', 'Monitor', 'Desk phone', 
                         'Ergonomic chair'],
            'Training': ['Professional certification', 'Conference registration', 
                        'Online course', 'Workshop attendance'],
            'Software': ['License renewal', 'Software subscription', 
                        'Development tools'],
            'Consulting': ['Strategy consulting', 'Technical advisory', 
                          'Legal services', 'Audit services']
        }
        
        return np.random.choice(descriptions.get(category, ['Business expense']))